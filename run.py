import logging

import controller


def main():
    logging.basicConfig(level=logging.DEBUG)

    t = controller.TankController()
    t.run_random()


if __name__ == '__main__':
    main()
