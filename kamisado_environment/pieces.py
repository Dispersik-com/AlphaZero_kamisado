
class Monk:
    def __init__(self, command_color: str, self_color: str):
        self.command_color = command_color
        self.self_color = self_color

    def __str__(self):
        return f"{self.command_color[0]}-{self.self_color}"

    def __repr__(self) -> str:
        return f"{self.command_color[0]}-{self.self_color}"
