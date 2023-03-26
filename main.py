from argparse import ArgumentParser

from util import FractalClass
from util import VisualizationMode
from visualize import ZoomAutoVisualizer
from visualize import ZoomClickVisualizer


def main():
    parser = ArgumentParser()
    parser.add_argument('--mode', type=VisualizationMode, default=VisualizationMode.AUTO)
    parser.add_argument('--fractal', type=FractalClass, default=FractalClass.MANDELBROT)
    parser.add_argument('--resolution', type=int, default=400)
    parser.add_argument('--zoom_fac', type=float, default=1.03)
    args = parser.parse_args()

    if args.mode == VisualizationMode.AUTO:
        visualizer = ZoomAutoVisualizer(fractal=args.fractal, resolution=args.resolution, zoom_fac=args.zoom_fac)
    elif args.mode == VisualizationMode.CLICK:
        visualizer = ZoomClickVisualizer(fractal=args.fractal, resolution=args.resolution, zoom_fac=args.zoom_fac)
    else:
        raise NotImplementedError(f'Unsupported visualization mode {args.mode}')

    visualizer.visualize()


if __name__ == '__main__':
    main()
