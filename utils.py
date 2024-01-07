import random
import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.Qt import *
from conversational_agent import *
from get_data import *
import time

history_msgs = None
user_all_msgs_history_msgs = None


def get_entity_message(name, text):
    return get_entity(name) + text


def get_entity(name, is_bot=False):
    return ("[USER] " if not is_bot else "[AGENT] ") + name + ": "


def show_alert(title, text):
    alert = QMessageBox()
    alert.setIcon(QMessageBox.Warning)
    alert.setText(text)
    alert.setWindowTitle(title)
    alert.exec_()


def get_input_dialog(window, title, label, resized=True):
    input_dialog = QInputDialog()
    if not resized:
        user_name, name_ok = input_dialog.getText(window, title, label)
    else:
        input_dialog.setInputMode(QInputDialog.TextInput)
        input_dialog.setWindowTitle(title)
        input_dialog.setLabelText(label)
        input_dialog.resize(225, 100)
        name_ok = input_dialog.exec_()
        user_name = input_dialog.textValue()

    return user_name, name_ok


def get_user_name(window):
    user_name, name_ok = get_input_dialog(window, title='Enter user name', label='Enter your name:')

    if not name_ok:
        return user_name, name_ok
    elif len(user_name) < 1:
        show_alert("Wrong name", "Your name must have at least one character!")

        return get_user_name(window), name_ok

    return user_name, name_ok


def get_agent_name(window):
    agent_name, name_ok = get_input_dialog(window, title='Enter agent name', label='Enter the name of the agent:')

    if not name_ok:
        return agent_name, name_ok
    elif len(agent_name) < 1:
        show_alert("Wrong name", "Your name must have at least one character!")

        return get_agent_name(window), name_ok

    return agent_name, name_ok


def chat_window():
    global history_msgs, user_all_msgs_history_msgs
    print("[Preparing the chat room...]")
    app = QApplication([])
    text_area = QTextEdit()
    text_area.setFocusPolicy(Qt.NoFocus)
    message = QLineEdit()
    message.setPlaceholderText("Type a message here")
    p = message.palette()
    p.setColor(message.foregroundRole(), Qt.green)
    p.setColor(message.backgroundRole(), Qt.black)
    message.setPalette(p)
    layout = QVBoxLayout()
    layout.addWidget(text_area)
    layout.addWidget(message)
    window = QWidget()
    window.setFixedWidth(500)
    window.setFixedHeight(750)
    p = window.palette()
    p.setColor(window.backgroundRole(), Qt.black)
    window.setPalette(p)
    window.setLayout(layout)
    window.setWindowTitle("Historical Conversational Agent Chat Room")
    window.show()

    user_name, name_ok = get_user_name(window)
    agent_name, name_ok = get_agent_name(window)

    if not name_ok:
        return

    agent_name, sentences, page_content = read_data_from_wiki(agent_name)
    model_name = "multi-qa-mpnet-base-cos-v1"
    bi_encoder = SentenceTransformer(model_name)
    if torch.cuda.is_available():
        bi_encoder = bi_encoder.to(torch.device("cuda"))
    paragraphs = get_paragraphs(page_content)
    paragraphs_embeddings = bi_encoder.encode(paragraphs, convert_to_tensor=True,
                                              show_progress_bar=True)

    intents, patterns = get_intents()
    patterns_embeddings = bi_encoder.encode(patterns, convert_to_tensor=True, show_progress_bar=True)

    agent = ConversationalAgent(agent_name, page_content)

    font, is_font_set = QFontDialog.getFont()
    bye_msgs = [str(msg).lower() for msg in
                ["Bye", "See you later", "Goodbye", "Nice chatting to you, bye", "Till next time"]]
    repeated_question_msgs = [str(msg).lower() for msg in
                              ["I just told you!", "You just asked me that before...", "Why do you ask me that again?"]]
    user_history = []
    user_all_msgs_history = []

    if is_font_set:
        text_area.setFont(font)
        message.setFont(font)

        def send_user_message():
            global history_msgs, user_all_msgs_history_msgs
            user_message = message.text().lower()

            if history_msgs is None:
                history_msgs = ""
            if user_all_msgs_history_msgs is None:
                user_all_msgs_history_msgs = ""

            text_area.moveCursor(QTextCursor.StartOfLine)
            text_area.setTextColor(Qt.green)
            text_area.insertPlainText(get_entity(user_name))

            text_area.moveCursor(QTextCursor.EndOfLine)
            text_area.setTextColor(Qt.black)
            text_area.insertPlainText(user_message)

            text_area.append("")

            message.clear()

            # get agent response afterwards
            text_area.moveCursor(QTextCursor.StartOfLine)
            text_area.setTextColor(Qt.red)
            text_area.insertPlainText(get_entity(agent.name, is_bot=True))

            text_area.moveCursor(QTextCursor.EndOfLine)
            text_area.setTextColor(Qt.black)

            # if the message is not relevant to the agent's current context
            # get a score of user_message related to the context
            if len(user_all_msgs_history) > 0 and (
                    user_all_msgs_history_msgs is not None and user_all_msgs_history_msgs != ""):
                score3 = is_relevant_question(user_message, user_all_msgs_history_msgs, bi_encoder=bi_encoder,
                                              is_history=True)
            else:
                score3 = 0

            """
            elif (score1 < 0.25 and score2 > 0.5) or "i" in user_message:
                response = get_response_by_question(intents, patterns, user_message, bi_encoder=bi_encoder,
                                                    patterns_embeddings=patterns_embeddings)
                text_area.insertPlainText(response)
            """

            if score3 >= 0.95:
                response = repeated_question_msgs[random.randint(0, len(repeated_question_msgs) - 1)]
                text_area.insertPlainText(response)
            else:
                score2 = is_relevant_question(user_message, patterns, bi_encoder=bi_encoder,
                                              paragraphs_embeddings=patterns_embeddings)
                if score2 >= 0.75:
                    response = get_response_by_question(intents, patterns, user_message, bi_encoder=bi_encoder,
                                                        patterns_embeddings=patterns_embeddings)
                    text_area.insertPlainText(response)
                elif score2 < 0.75:
                    if len(user_history) > 0:
                        score1 = is_relevant_question(user_message, history_msgs, is_history=True,
                                                      bi_encoder=bi_encoder)
                    else:
                        score1 = 0

                    score4 = is_relevant_question(user_message, page_content, bi_encoder=bi_encoder, paragraphs_embeddings=paragraphs_embeddings)
                    if score1 > score4:
                        if score1 < 0.15:
                            text_area.insertPlainText(agent.get_gen_response(user_message))
                        elif score1 > 0.5:
                            context = get_context_based_on_question2(user_message, history_msgs, top_k=5, is_history=True,
                                                                     bi_encoder=bi_encoder,
                                                                     paragraphs_embeddings=None)
                            agent.biography = context
                            text_area.insertPlainText(agent.get_response(user_message))
                        else:
                            context = get_context_based_on_question2(user_message, page_content, top_k=5,
                                                                     bi_encoder=bi_encoder,
                                                                     paragraphs_embeddings=paragraphs_embeddings)
                            agent.biography = context
                            text_area.insertPlainText(agent.get_response(user_message))
                    else:
                        if score4 < 0.15:
                            text_area.insertPlainText(agent.get_gen_response(user_message))
                        else:
                            context = get_context_based_on_question2(user_message, page_content, top_k=5,
                                                                     bi_encoder=bi_encoder,
                                                                     paragraphs_embeddings=paragraphs_embeddings)
                            agent.biography = context
                            text_area.insertPlainText(agent.get_response(user_message))

            text_area.append("")
            text_area.append("")

            if "?" not in user_message:
                user_history.append(user_message)

            user_all_msgs_history.append(user_message)

            # add chat history
            if len(user_history) > 0:
                history_msgs = ""
                for msg in user_history:
                    history_msgs += msg + "\n"

            if len(user_all_msgs_history) > 0:
                user_all_msgs_history_msgs = ""
                for msg in user_all_msgs_history:
                    user_all_msgs_history_msgs += msg + "\n"

            """if user_message == "exit" or user_message == "quit" or user_message in bye_msgs:
                time.sleep(1)
                sys.exit(0)"""

        message.returnPressed.connect(send_user_message)

        sys.exit(app.exec_())
