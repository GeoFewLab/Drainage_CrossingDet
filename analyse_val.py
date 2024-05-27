import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple
from scipy.spatial.distance import euclidean
from tqdm import tqdm
from ast import literal_eval  # Importing literal_eval

def analyse_val():
    Rectangle = namedtuple('Rectangle', ['xmin', 'ymin', 'xmax', 'ymax'])

    val_res = pd.read_csv('./validation_results/validation_results.csv',index_col=False)
    labels = pd.read_csv('./validation_results/val_labels.csv')
    
    labels = labels.groupby(['filename']).agg(tuple).applymap(list).reset_index()
    
    # Calculate total_true_labels correctly
    total_true_labels = labels['box_type'].apply(lambda x: x.count(True)).sum()
    
    def convert_to_literal(row):
        label_list = literal_eval(row['labels'])
        centroid_list = literal_eval(row['centroids'])
        x1_list = literal_eval(row['x1'])
        y1_list = literal_eval(row['y1'])
        x2_list = literal_eval(row['x2'])
        y2_list = literal_eval(row['y2'])

        bounding_boxes = [Rectangle(x1, y1, x2, y2) for x1, y1, x2, y2 in zip(x1_list, y1_list, x2_list, y2_list)]
        return label_list, centroid_list, bounding_boxes

    def check_overlap(a, b):
        ''' 
        Calculate the intersection over union (IoU) between two rectangles.

        Parameters:
            a (Rectangle): First rectangle.
            b (Rectangle): Second rectangle.

        Returns:
            float: Intersection over union (IoU) value.
        '''
        # Determine the (x, y)-coordinates of the intersection rectangle
        xA = max(a.xmin, b.xmin)
        yA = max(a.ymin, b.ymin)
        xB = min(a.xmax, b.xmax)
        yB = min(a.ymax, b.ymax)

        # Compute the area of intersection rectangle
        interArea = max(xB - xA, 0) * max(yB - yA, 0)

        # Compute the area of both the prediction and ground-truth rectangles
        boxAArea = abs((a.xmax - a.xmin) * (a.ymax - a.ymin))
        boxBArea = abs((b.xmax - b.xmin) * (b.ymax - b.ymin))

        # Compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # Return the intersection over union value
        return iou

    def find_centroid(a):
        return ((a.xmax + a.xmin) / 2, (a.ymin + a.ymax) / 2)
    
    def closest_center(centroids, point):
        if not centroids:
            return -1
        distances = [euclidean(c, point) for c in centroids]
        return distances.index(min(distances))

    
    overlap_vals, labels_missed, extra_labels = [], 0, 0

    print("Running Analysis:")
    for index, row in tqdm(val_res.iterrows(), total=val_res.shape[0]):
        cur_label_row = labels.iloc[index]
        label_bounds = [Rectangle(xmin, ymin, xmax, ymax) for xmin, ymin, xmax, ymax in zip(cur_label_row['xmin'], cur_label_row['ymin'], cur_label_row['xmax'], cur_label_row['ymax'])]
        label_centroids = [find_centroid(box) for box in label_bounds]
        _, cur_centroids, bounding_boxes = convert_to_literal(row)

        # Mapping and efficiency improvement
        prediction_mapping = [closest_center(cur_centroids, center) for center in label_centroids]

        # Calculate extra labels and missed labels
        matched_predictions = set(prediction_mapping)
        extra_labels += len(set(range(len(cur_centroids))) - matched_predictions)
        labels_missed += prediction_mapping.count(-1)

        # Calculate overlaps
        overlaps = [check_overlap(label_bounds[i], bounding_boxes[prediction_mapping[i]]) for i in range(len(label_bounds)) if prediction_mapping[i] != -1]
        overlap_vals.extend(overlaps)
    print(len(overlap_vals))
    print("Done!")
    avg_overlap = round(sum(overlap_vals) / len(overlap_vals) if overlap_vals else 0, 2)
    precision = round(len([ov for ov in overlap_vals if ov > 0]) / (len(overlap_vals) + extra_labels) if overlap_vals else 0, 2)
    recall = round(len([ov for ov in overlap_vals if ov > 0]) / total_true_labels, 2)
    f1 = round(2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0, 2)
    
    # Outputting the evaluation metrics
    print(f"Total True Labels: {total_true_labels}")
    print(f"Labels Missed: {labels_missed}")
    print(f"Extra Labels: {extra_labels}")
    print(f"Average Overlap: {avg_overlap}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    # Visualizing the evaluation metrics
    metrics_data = [
        ["Metric", "Value"],
        ["Average Overlap", avg_overlap],
        ["Precision", precision],
        ["Recall", recall],
        ["F1 Score", f1]
    
    ]

    fig, ax = plt.subplots(figsize=(5, 4))  # Adjusted for better visualization
    table = ax.table(cellText=metrics_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # Adjust scale for better readability
    ax.axis('off')
    plt.title('Validation Evaluation Metrics')
    plt.savefig('./validation_results/metrics_table.png', bbox_inches='tight')  # Save as an image
    plt.show()
    plt.show()


analyse_val()