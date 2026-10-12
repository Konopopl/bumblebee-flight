def pytest_addoption(parser):
    parser.addoption(
        "--regression-output",
        help="Directory containing results.npz and config.json from a completed full run",
    )
