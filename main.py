from argparse import ArgumentParser

from util import VisualizationMode
from visualize import ZoomAutoVisualizer
from visualize import ZoomClickVisualizer


def main():
    parser = ArgumentParser()
    parser.add_argument('--mode', type=VisualizationMode, default=VisualizationMode.AUTO)
    parser.add_argument('--resolution', type=int, default=200)
    parser.add_argument('--zoom_fac', type=float, default=1.05)
    args = parser.parse_args()

    if args.mode == VisualizationMode.AUTO:
        visualizer = ZoomAutoVisualizer(resolution=args.resolution, zoom_fac=args.zoom_fac)
    elif args.mode == VisualizationMode.CLICK:
        visualizer = ZoomClickVisualizer(resolution=args.resolution, zoom_fac=args.zoom_fac)
    else:
        raise NotImplementedError(f'Unsupported visualization mode {args.mode}')

    visualizer.visualize()


if __name__ == '__main__':
    main()
