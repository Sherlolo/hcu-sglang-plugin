from hcu_sglang.compat.aiter import bootstrap_aiter_compat

bootstrap_aiter_compat()


def main():
    import runpy

    runpy.run_module('sglang.launch_server', run_name='__main__', alter_sys=False)


if __name__ == '__main__':
    main()
