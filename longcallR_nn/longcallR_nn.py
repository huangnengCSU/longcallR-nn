import sys
import argparse
from ._version import __version__
from .train import train_process
from .predict import call_process
from .download import download_configs_and_models


def main():
    parser = argparse.ArgumentParser(
        description='longcallR_nn: a deep learning based variant caller for long-reads RNA-seq data')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s {version}'.format(version=__version__))
    subparser = parser.add_subparsers(dest='command', help='longcallR_nn commands', required=True)

    ### sub-command train
    train_parser = subparser.add_parser('train', help='train model')
    train_parser.add_argument('-config', type=str, help='path to config file', required=True)
    train_parser.add_argument('-log', type=str, default='train.log', help='name of log file')
    train_parser.set_defaults(func=train_process)

    ### sub-command call
    call_parser = subparser.add_parser('call', help='call variants')
    call_parser.add_argument('-config', type=str, required=True, help='path to config file')
    call_parser.add_argument('-model', required=True, help='path to trained model')
    call_parser.add_argument('-data', required=True, help='directory of feature files')
    call_parser.add_argument('-ref', required=False, default=None, help='reference genome file')
    call_parser.add_argument('-output', required=True, help='output vcf file')
    call_parser.add_argument('-max_depth', type=int, default=2000, help='max depth threshold')
    call_parser.add_argument('-batch_size', type=int, default=1000, help='batch size')
    call_parser.add_argument('--no_cuda', action="store_true", help='If running on cpu device, set the argument.')
    call_parser.set_defaults(func=call_process)

    ### sub-command download
    download_parser = subparser.add_parser('download', help='download configuration files and models')
    download_parser.add_argument('-d', '--download_dir', type=str, help='download directory', default='models')
    download_parser.set_defaults(func=download_configs_and_models)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
