import argparse
import logging

import controller


def main(dry_run):
    logging.basicConfig(level=logging.INFO)

    t = controller.TankController(dry_run)
    t.run_look_at_person()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dry-run',
        dest='dry_run',
        action='store_true',
        help=(
            'Run without connection to the Octopi API, '
            'tank commands are only logged but not submitted.'
        )
    )
    parser.set_defaults(dry_run=False)
    return parser.parse_args()


if __name__ == '__main__':
    arguments = get_arguments()
    main(arguments.dry_run)
