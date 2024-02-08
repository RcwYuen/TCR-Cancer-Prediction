import re
import matplotlib.pyplot as plt
import time

# File to monitor
filename = 'training-logs.log'

# Regex to match both line formats and extract the numerical value
pattern_avg_loss = re.compile(r"\[.+\]: Accummulated Loss \(Average\): (\d+\.\d+)")
pattern_file_loss = re.compile(r"\[.+\]: File \d+ / \d+: Loss: (\d+\.\d+)")

def extract_losses(filename):
    """Extract and return lists of accumulated average losses and file-specific losses."""
    avg_losses = []
    file_losses = []

    with open(filename, 'r') as file:
        for line in file:
            match_avg = pattern_avg_loss.search(line)
            match_file = pattern_file_loss.search(line)
            if match_avg:
                avg_losses.append(float(match_avg.group(1)))
            elif match_file:
                file_losses.append(float(match_file.group(1)))
                
    return avg_losses, file_losses

def create_plot(filename):
    avg_losses, file_losses = extract_losses(filename)
    fig, axs = plt.subplots(ncols = 2)
    axs[0].scatter([i for i in range(len(avg_losses))], avg_losses)
    axs[0].set_title("Accummulated Average Loss")
    axs[1].scatter([i for i in range(len(file_losses))], file_losses)
    axs[1].set_title("Loss by File")
    plt.show()

if __name__ == "__main__":
    create_plot(filename)