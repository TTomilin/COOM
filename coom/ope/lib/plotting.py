import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter1d


def plot_line(file_name: str, style: str, color: str = None):
    file = os.path.join(log_dir, f'{args.previous_domain}_{file_name}.csv')
    df = pd.read_csv(file)
    y = df.iloc[:, 1]
    y = gaussian_filter1d(y, sigma=2)
    plot = plt.plot(y, label=file_name, linestyle=style, color=color)
    return plot[0].get_color()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-d', default="data/wm", help='The base data/output directory')
    parser.add_argument('--previous_domain', default=None, type=str, choices=['default', 'B1', 'B2', 'B3'],
                        help='Predefined colors of the previously trained Car Racing environment')
    parser.add_argument('--domains', default=['default'], type=str, nargs="+", choices=['default', 'B1', 'B2', 'B3'],
                        help='Use predefined colors for the Car Racing environment')
    parser.add_argument('--colors', default=['blue', 'red', 'green'], type=str, nargs="+", help='Colors for the plots')
    parser.add_argument('--files', default=[], type=str, nargs="+", help='The files to plot')
    parser.add_argument('--game', default='CarRacing-v2', help='Game to use')
    parser.add_argument('--experiment_name', default='experiment_1', help='To isolate its files from others')
    parser.add_argument('--model_type', default='controller', choices=['vision', 'model', 'controller'])
    parser.add_argument('--title', type=str, default='Continual Learning')
    parser.add_argument('--x_label', type=str, default='Steps')
    parser.add_argument('--y_label', type=str, default='Reward')

    args = parser.parse_args()

    ope_dir = Path(__file__).parent.parent.resolve()
    log_dir = os.path.join(ope_dir, args.data_dir, args.game, args.experiment_name, args.model_type, 'logs')

    plt.style.use('ggplot')
    for domain, color in zip(args.domains, args.colors):
        color = plot_line(domain, 'solid')
        plot_line(f'{domain}_retrained', 'dashed', color)

    plt.title(args.title)
    plt.xlabel(args.x_label)
    plt.ylabel(args.y_label)
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'CL_comparison.png'))
    plt.show()
