import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


def compute_average_similarity(detectors, seqs, conf=0.5):
    for idx, detector in enumerate(detectors):
        for seq in seqs:
            det = np.load(f"./dets/{detector}/{seq}.npz")
            nframes = int(len(det.files) / 2)
            fidx = 0
            seq_mean = 0
            seq_var = 0
            for frame in range(nframes):
                bbox = det[f'{frame}_det']
                reid_vectors = det[f'{frame}_feat'][bbox[:, 4] > conf]
                if reid_vectors.shape[0] <= 1:
                    continue
                similarity_matrix = cosine_similarity(reid_vectors)
                # Extract upper triangle of similarity matrix (excluding diagonal)
                upper_tri = similarity_matrix[np.triu_indices(len(reid_vectors), k=1)]
                seq_mean += np.mean(upper_tri)  # Mean pairwise similarity
                seq_var += np.std(upper_tri)  # Variance of similarity scores
                fidx += 1
            seq_mean /= fidx
            seq_var /= fidx
            print(detector, seq, seq_mean, seq_var)


def visual_similarity(frame=100, conf=0.3):
    # Create a 4x4 grid of subplots
    fig, axes = plt.subplots(3, 4, figsize=(20, 20))  # Adjust figsize for clarity
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    # Iterate over detectors and populate the heatmaps
    detectors = ['detector_poi', 'detector_trades', 'detectors_yolov11', 'detector_jde',
                 'detector_cstrack', 'detector_fairmot128', 'detector_gsdt', 'detector_bytetrack']
    detector_names = ['POI', 'TraDes', 'YOLOv11_SBS50', 'JDE', 'CSTrack', 'FairMOT', 'GSDT', 'YOLOX_SBS50']
    seq = "MOT16-04"
    for idx, detector in enumerate(detectors):
        det = np.load(f"./dets/{detector}/{seq}.npz")
        # Step 1: Example ReID vectors (replace with your actual data for each detector)
        bbox = det[f'{frame}_det']
        reid_vectors = det[f'{frame}_feat'][bbox[:, 4] > conf]
        # Step 2: Compute the similarity matrix
        similarity_matrix = cosine_similarity(reid_vectors)
        # Step 3: Visualize the heatmap in the corresponding subplot
        sns.heatmap(similarity_matrix, annot=False, cmap="YlGnBu", vmin=0, vmax=1, ax=axes[idx])
        axes[idx].set_title(f"{seq}, {detector_names[idx]}")
        # axes[idx].set_xlabel("Vector Index")
        # axes[idx].set_ylabel("Vector Index")
        # Step 4: Set discrete ticks (e.g., every 2nd index)
        tick_indices = np.arange(0, len(reid_vectors), 2)  # Show every 2nd index (0, 2, 4, 6, 8)
        axes[idx].set_xticks(tick_indices)
        axes[idx].set_yticks(tick_indices)

    detectors_mot20 = ['detector_cstrack', 'detector_fairmot128', 'detector_gsdt', 'detector_bytetrack']
    detectors_mot20_names = ['CSTrack', 'FairMOT', 'GSDT', 'YOLOX_SBS50']
    seq = "MOT20-05"
    for idx, detector in enumerate(detectors_mot20):
        det = np.load(f"./dets/{detector}/{seq}.npz")
        # Step 1: Example ReID vectors (replace with your actual data for each detector)
        bbox = det[f'{frame}_det']
        reid_vectors = det[f'{frame}_feat'][bbox[:, 4] > conf]
        # Step 2: Compute the similarity matrix
        similarity_matrix = cosine_similarity(reid_vectors)
        # Step 3: Visualize the heatmap in the corresponding subplot
        sns.heatmap(similarity_matrix, annot=False, cmap="YlGnBu", vmin=0, vmax=1, ax=axes[idx + len(detectors)])
        axes[idx + len(detectors)].set_title(f"{seq}, {detectors_mot20_names[idx]}")
        # axes[idx].set_xlabel("Vector Index")
        # axes[idx].set_ylabel("Vector Index")
        # Step 4: Set discrete ticks (e.g., every 2nd index)
        tick_indices = np.arange(0, len(reid_vectors), 2)  # Show every 2nd index (0, 2, 4, 6, 8)
        axes[idx + len(detectors)].set_xticks(tick_indices)
        axes[idx + len(detectors)].set_yticks(tick_indices)
    # Hide any unused subplots (if fewer than 16 detectors)
    for idx in range(len(detectors) + len(detectors_mot20), len(axes)):
        axes[idx].axis('off')
    # Adjust layout to prevent overlap
    # plt.subplots_adjust(hspace=5.5)  # Default is ~0.2; increase for wider spacing
    plt.tight_layout(h_pad=5.0, rect=[0, 0, 1, 0.97])
    # Save the figure as a PDF file
    plt.savefig("reid_similarity_heatmaps.pdf", format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    visual_similarity()

    detectors = ['detector_poi', 'detector_trades', 'detectors_yolov11', 'detector_jde',
                 'detector_cstrack', 'detector_fairmot128', 'detector_gsdt', 'detector_bytetrack']
    seqs = ['MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-09', 'MOT16-10', 'MOT16-11', 'MOT16-13']
    detectors_mot20 = ['detector_cstrack', 'detector_fairmot128', 'detector_gsdt', 'detector_bytetrack']
    seqs_mot20 = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05']
    compute_average_similarity(detectors, seqs)
    compute_average_similarity(detectors_mot20, seqs_mot20)
