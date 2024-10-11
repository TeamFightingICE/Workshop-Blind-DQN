import asyncio

import typer
from loguru import logger
from typing_extensions import Annotated, Optional

from src.train.train import run_train
from src.test.test import run_test

app = typer.Typer()

@app.command()
def train(
        game_num: Annotated[int, typer.Option(help="Number of games to play per iteration")] = 5,
        port: Annotated[Optional[int], typer.Option(help="Port used by DareFightingICE")] = 31415):
    asyncio.run(run_train(game_num, port))


@app.command()
def test(
    game_num: Annotated[int, typer.Option(help="Number of games to play per iteration")] = 5,
        port: Annotated[Optional[int], typer.Option(help="Port used by DareFightingICE")] = 31415):
    asyncio.run(run_test(game_num, port))


if __name__ == '__main__':
    app()