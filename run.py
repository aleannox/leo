import logging

import controller


def main():
    logging.basicConfig(level=logging.INFO)

    t = controller.TankController()
    # t.run_random()
    t.run_look_at_person()


if __name__ == '__main__':
    main()
