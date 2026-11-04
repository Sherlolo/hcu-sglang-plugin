# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

def main():
    import runpy

    runpy.run_module('hcu_sglang.launch_server', run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    main()
