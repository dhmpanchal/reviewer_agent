
class Utils:
    def __init__(self) -> None:
        pass

    def read_from_file(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content

    def write_to_file(self, file_path: str, content: str):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Content written to {file_path}")