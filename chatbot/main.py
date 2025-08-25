import uuid
from termcolor import cprint

from . import config
from .bot import get_rag_chain

def main():
    session_id = str(uuid.uuid4())
    rag_chain = get_rag_chain()

    bot_name = config.BOT_NAME
    cprint(f"[{bot_name}]: {config.BOT_GREETING_MESSAGE}", "green")

    while True:
        try:
            cprint("[YOU]: ", "blue", attrs=["bold"], end="")
            user_input = input()
            if user_input.lower() in ["exit", "quit"]:
                break

            response = rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )
            cprint(f"[{bot_name}]: {response.content}", "green")

        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    main()
