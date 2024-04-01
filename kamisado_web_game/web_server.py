import base64
from aiohttp import web
import json
import asyncio
from game_environment.kamisado_enviroment import KamisadoGame
# from MCTS.neural_mcts import NeuralMonteCarloTreeSearch


class GameServer:

    def __init__(self,  ip_addr, port):
        self.ip_addr = ip_addr
        self.port = port
        self.game = KamisadoGame()

    def reload_game(self):
        self.game = KamisadoGame()

    async def serve_game_page(self, request):
        try:
            with open('web_game.html', 'r') as html_file:
                html_content = html_file.read()

            with open('styles.css', 'r') as css_file:
                css_content = css_file.read()

            with open('script.js', 'r') as js_file:
                js_content = js_file.read()

            # with open('monk.png', 'rb') as image_file:
            #     image_data = image_file.read()
            #     image_base64 = base64.b64encode(image_data).decode('utf-8')

            combined_content = f"""
                    <!DOCTYPE html>
                    <html lang="en">
                    <head>
                      <meta charset="UTF-8">
                      <meta name="viewport" content="width=device-width, initial-scale=1.0">
                      <title>Kamisado Game</title>
                    </head>
                    <body>
                        <style>{css_content}</style>
                        <script>{js_content}</script>
                    </head>
                    <body>
                        {html_content}
                    </body>
                    </html>
                """

            self.reload_game()
            return web.Response(text=combined_content, content_type='text/html')
        except FileNotFoundError:
            return web.Response(status=404)

    async def handle(self, request):
        data = await request.json()
        start_cell = data["last_cell"]["row"], data["last_cell"]["col"]
        end_cell = data["new_cell"]["row"], data["new_cell"]["col"]
        self.game.make_move(start_cell, end_cell)

        response = {
                    "legal_moves": self.game.get_legal_moves(),
                    "winner": self.game.check_winner(),
                    }
        return web.json_response(response,
                                 headers={'Access-Control-Allow-Origin': f'http://{self.ip_addr}:{self.port}'})

    async def start_server(self):
        app = web.Application()
        app.router.add_post('/handle', self.handle)
        app.router.add_get('/game', self.serve_game_page)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.ip_addr, self.port)
        await site.start()
        print(f"Server started at http://{self.ip_addr}:{self.port}")
        await asyncio.Event().wait()

    async def run(self):
        await self.start_server()


if __name__ == "__main__":
    server = GameServer("localhost", 8080)
    asyncio.run(server.run())
