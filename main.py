#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import getopt


def print_usage():
    # TODO write usage text here
    pass


def main():
    if len(sys.argv) < 2:  # TODO change number of program arguments here
        print_usage()
        sys.exit(2)
    try:
        options, _ = getopt.getopt(sys.argv[1:], "c:")  # TODO maybe some changes here
    except getopt.GetoptError as err:
        print(str(err))
        print_usage()
        sys.exit(2)
    for o, a in options:    # TODO change here the program arguments (also the optional ones)
        if o == "-c":
            pass


if __name__ == '__main__':
    main()
