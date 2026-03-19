import matplotlib.pyplot as plt
import torch
import random
import numpy as np
import os
import sys
import errno


def setup_system(seed, cudnn_benchmark=True, cudnn_deterministic=True) -> None:
    '''
    Set seeds for for reproducible training
    '''
    # python
    random.seed(seed)

    # numpy
    np.random.seed(seed)

    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn_benchmark_enabled = cudnn_benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w', encoding='utf-8')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class LossLogger:
    def __init__(self, log_dir="./university"):
        """
        Loss logger

        Args:
            log_dir (str): Directory to save logs
        """
        self.log_dir = log_dir

        # Initialize storage variables
        self.iter_count = 0       # Total iteration counter
        self.current_epoch = 1    # Current epoch
        self.global_loss_sum = 0.0     # Global accumulated loss
        self.global_avg_loss = 0.0      # Global average loss

        # Store only epoch-level data
        self.epoch_losses = []            # Average loss for each epoch
        self.epoch_global_avg_losses = [] # Global average loss at the end of each epoch
        self.epoch_batch_counts = []      # Number of batches per epoch

        # Temporary data for the current epoch
        self.current_epoch_loss_sum = 0.0
        self.current_epoch_batch_count = 0
        self.epoch_avg_loss = 0.0

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # Initialize log file
        with open(os.path.join(self.log_dir, "loss_log.txt"), "w") as f:
            f.write(f"{'Epoch':^7} {'Batches':^9} {'Epoch Loss':^12} {'Global Avg':^12}\n")

    def reset(self):
        """Reset all states"""
        self.iter_count = 0
        self.current_epoch = 1
        self.global_loss_sum = 0.0
        self.global_avg_loss = 0.0

        # Reset epoch-level data
        self.epoch_losses = []
        self.epoch_global_avg_losses = []
        self.epoch_batch_counts = []

        # Reset temporary epoch data
        self.current_epoch_loss_sum = 0.0
        self.current_epoch_batch_count = 0
        self.epoch_avg_loss = 0.0

    def update(self, epoch, loss):
        """
        Record the loss of a single iteration

        Args:
            epoch (int): Current epoch index
            loss (float): Loss value of the current iteration
        """

        # Update iteration counter
        self.iter_count += 1
        self.global_loss_sum += loss

        # Update global average loss
        self.global_avg_loss = self.global_loss_sum / self.iter_count

        # Check if a new epoch has started
        if epoch != self.current_epoch:
            # Finish the previous epoch
            self.end_epoch()
            # Start a new epoch
            self.current_epoch = epoch

        # Update loss sum and batch count for the current epoch
        self.current_epoch_loss_sum += loss
        self.current_epoch_batch_count += 1

    def end_epoch(self):
        """Mark the end of the current epoch and save its loss"""
        if self.current_epoch_batch_count == 0:
            return  # No data, skip

        # Compute average loss for this epoch
        self.epoch_avg_loss = self.current_epoch_loss_sum / self.current_epoch_batch_count

        # Save epoch data
        self.epoch_losses.append(self.epoch_avg_loss)
        self.epoch_global_avg_losses.append(self.global_avg_loss)
        self.epoch_batch_counts.append(self.current_epoch_batch_count)

        # Write to log file
        with open(os.path.join(self.log_dir, "loss_log.txt"), "a") as f:
            f.write(f"{self.current_epoch:^7} "
                    f"{self.current_epoch_batch_count:^9} "
                    f"{self.epoch_avg_loss:^12.6f} "
                    f"{self.global_avg_loss:^12.6f}\n")

        # Reset temporary epoch data for the next epoch
        self.current_epoch_loss_sum = 0.0
        self.current_epoch_batch_count = 0

    def final_plot(self):
        """Save the final loss curve after training finishes"""
        if not self.epoch_losses:
            print("No epoch data to plot.")
            return

        plt.figure(figsize=(14, 8))

        # Create epoch indices
        epochs = np.arange(1, len(self.epoch_losses) + 1)
        num_epochs = len(self.epoch_losses)

        # Plot average loss per epoch
        plt.plot(epochs, self.epoch_losses, 'b-', linewidth=2, label='Epoch Average Loss')

        # Plot global average loss per epoch
        plt.plot(epochs, self.epoch_global_avg_losses, 'r-', linewidth=2, label='Global Average Loss')

        # Titles and labels
        plt.title('Training Loss per Epoch', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Legend
        plt.legend(fontsize=12)

        # Smart x-tick management
        if num_epochs <= 20:
            plt.xticks(epochs)
        elif num_epochs <= 50:
            step = 2
            ticks = np.arange(1, num_epochs + 1, step)
            if num_epochs not in ticks:
                ticks = np.append(ticks, num_epochs)
            plt.xticks(ticks)
        else:
            step = 5
            ticks = np.arange(1, num_epochs + 1, step)
            if num_epochs not in ticks:
                ticks = np.append(ticks, num_epochs)
            plt.xticks(ticks)

        # Save image
        plt.savefig(os.path.join(self.log_dir, "loss_curve.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def get_summary(self):
        """Return training summary"""
        if not self.epoch_losses:
            return {}

        return {
            "total_epochs": len(self.epoch_losses),
            "total_iterations": self.iter_count,
            "final_epoch_loss": self.epoch_losses[-1],
            "final_global_avg": self.epoch_global_avg_losses[-1],
            "best_epoch_loss": min(self.epoch_losses),
            "best_epoch": np.argmin(self.epoch_losses) + 1
        }

    def finalize(self):
        """Finish training and process the last epoch"""
        if self.current_epoch_batch_count > 0:
            self.end_epoch()

        # Save final loss curve
        self.final_plot()

        # Return training summary
        return self.get_summary()



# Example usage
if __name__ == "__main__":
    # Initialize loss logger
    logger = LossLogger(log_dir="./epoch_results")

    # Simulated training process - 3 epochs with different iteration counts
    epoch_iterations = [600, 300, 400]  # Number of iterations per epoch

    for epoch_idx, iterations in enumerate(epoch_iterations, 1):
        print(f"Starting Epoch {epoch_idx} with {iterations} iterations")

        for batch_idx in range(1, iterations + 1):
            # Simulate loss value
            progress = batch_idx / iterations
            base_loss = 0.5 * (1 - progress) + 0.1 * np.random.rand()
            loss = max(0.01, base_loss)

            # Record current iteration loss
            logger.update(epoch_idx, loss)

            # Periodically print progress
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch_idx}/{len(epoch_iterations)}], "
                      f"Batch [{batch_idx}/{iterations}], "
                      f"Loss: {loss:.6f}")

        # Mark end of current epoch
        logger.end_epoch()
        print(f"Epoch {epoch_idx} completed!")

    # Finish training and save results
    summary = logger.finalize()

    print("\nTraining complete!")
    print(f"Epoch loss log saved to: {os.path.join(logger.log_dir, 'loss_log.txt')}")
    print("Training summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")
