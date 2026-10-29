def main():
    import runpy

    runpy.run_module('hcu_sglang.launch_server', run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    main()
