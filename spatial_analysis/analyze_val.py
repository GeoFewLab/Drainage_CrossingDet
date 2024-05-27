from collections import namedtuple
from scipy.spatial.distance import euclidean
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyse_val():
    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

    # Load validation results and ground truth labels
    val_res = pd.read_csv('val_pts.csv')
    labels = pd.read_csv('NE_val_labels.csv')

    labels = labels.groupby(['filename']).agg(list).reset_index()
    val_res = val_res.groupby(['filename']).agg(list).reset_index()

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
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # Return the intersection over union value
        return iou

    def find_centroid(a):
        x = (a.xmax + a.xmin) / 2
        y = (a.ymax + a.ymin) / 2
        return (x, y)

    def closest_center(centroids, point):
        if not centroids:
            return -1
        distances = [euclidean(c, point) for c in centroids]
        return distances.index(min(distances))

    y_true = []
    y_pred = []
    overlap_vals = []
    labels_missed = 0
    extra_labels = 0
    total_true_labels = 0
    total_true_labels_corrected = sum(len(x) for x in labels['xmin'])
    print("Corrected total_true_labels:", total_true_labels_corrected)
    for index, row in val_res.iterrows():
        filename = row['filename']
     
        label_row_filtered = labels[labels['filename'] == filename]
        if label_row_filtered.empty:
             print(f"No matching label found for {filename}. Skipping.")
             continue  # Skip this iteration if no matching label found
        label_row = label_row_filtered.iloc[0]
        # print(f"Found matching label for {filename}. Processing...") 
        
        true_label_boxes = [Rectangle(xmin, ymin, xmax, ymax) for xmin, ymin, xmax, ymax in zip(label_row['xmin'], label_row['ymin'], label_row['xmax'], label_row['ymax'])]
        
        # print("Corrected total_true_labels:", total_true_labels_corrected)
        pred_boxes = [Rectangle(xmin, ymin, xmax, ymax) for xmin, ymin, xmax, ymax in zip(row['xmin'], row['ymin'], row['xmax'], row['ymax'])]

        label_centroids = [find_centroid(box) for box in true_label_boxes]
        pred_centroids = [find_centroid(box) for box in pred_boxes]

        matched_preds = []
        for label_centroid in label_centroids:
            closest_pred_index = closest_center(pred_centroids, label_centroid)
            if closest_pred_index != -1:
                matched_preds.append(pred_boxes.pop(closest_pred_index))
                pred_centroids.pop(closest_pred_index)
            else:
                labels_missed += 1

        extra_labels += len(pred_boxes)
        total_true_labels += len(true_label_boxes)
        
        overlaps = [check_overlap(true_box, pred_box) for true_box, pred_box in zip(true_label_boxes, matched_preds)]
        overlap_vals.extend(overlaps)

        # Update y_true and y_pred
        y_true.extend([1] * len(matched_preds) + [0] * len(pred_boxes))  # 1 for matched, 0 for extra preds
        y_pred.extend([1] * len(matched_preds) + [1] * len(pred_boxes))  # Predicted as 1 for both matched and extra

    avg_overlap = round(sum(overlap_vals) / len(overlap_vals) if overlap_vals else 0,2)
    precision = round(len([ov for ov in overlap_vals if ov > 0]) / (len(overlap_vals) + extra_labels) if overlap_vals else 0,2)
    recall = round(len([ov for ov in overlap_vals if ov > 0]) / total_true_labels_corrected,2)
    f1 = round(2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0,2)
    print("Extra_labels:", extra_labels)
    print("total_true_labels:", total_true_labels)
    print("len_overlap:", len(overlap_vals))

    metrics_data = [
        ['Average Overlap', avg_overlap],
        ['Precision', precision],
        ['Recall', recall],
        ['F1 Score', f1]
    ]

    plt.figure(figsize=(6, 4))
    plt.table(cellText=metrics_data, colLabels=['Metric', 'Value'], loc='center', cellLoc='center')
    plt.axis('off')  # Hide axes
    plt.title('Evaluation Metrics')
    plt.savefig('metrics_table.png', bbox_inches='tight')  
    plt.show()


analyse_val()