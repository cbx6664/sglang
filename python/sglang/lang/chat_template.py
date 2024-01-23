from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Dict, List, Tuple


class ChatTemplateStyle(Enum):
    PLAIN = auto()
    LLAMA2 = auto()


@dataclass
class ChatTemplate:
    name: str
    default_system_prompt: str
    role_prefix_and_suffix: Dict[str, Tuple[str]]
    image_token: str = "<image>"
    style: ChatTemplateStyle = ChatTemplateStyle.PLAIN

    def get_prefix_and_suffix(self, role, hist_messages):
        if self.style == ChatTemplateStyle.PLAIN:
            return self.role_prefix_and_suffix[role]
        elif self.style == ChatTemplateStyle.LLAMA2:
            if len(hist_messages) == 0 and role == "system":
                return (
                    self.role_prefix_and_suffix["user"][0]
                    + self.role_prefix_and_suffix["system"][0],
                    self.role_prefix_and_suffix["system"][1],
                )
            elif (
                len(hist_messages) == 1
                and role == "user"
                and hist_messages[0]["content"] is not None
            ):
                return ("", self.role_prefix_and_suffix["user"][1])
            return self.role_prefix_and_suffix[role]
        else:
            raise ValueError(f"Invalid style: {self.style}")

    def get_prompt(self, messages):
        prompt = ""
        for i in range(len(messages)):
            role, content = messages[i]["role"], messages[i]["content"]
            if role == "system" and content is None:
                content = self.default_system_prompt
                if content is None:
                    continue

            prefix, suffix = self.get_prefix_and_suffix(role, messages[:i])
            prompt += prefix + content + suffix
        return prompt


chat_template_registry: Dict[str, ChatTemplate] = {}
matching_function_registry: List[Callable] = []


def register_chat_template(template):
    chat_template_registry[template.name] = template


def register_chat_template_matching_function(func):
    matching_function_registry.append(func)


def get_chat_template(name):
    return chat_template_registry[name]


def get_chat_template_by_model_path(model_path):
    for matching_func in matching_function_registry:
        template = matching_func(model_path)
        if template is not None:
            return template
    return get_chat_template("default")


register_chat_template(
    ChatTemplate(
        name="default",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": ("SYSTEM:", "\n"),
            "user": ("USER:", "\n"),
            "assistant": ("ASSISTANT:", "\n"),
        },
    )
)


register_chat_template(
    ChatTemplate(
        name="claude",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": ("", ""),
            "user": ("\n\nHuman: ", ""),
            "assistant": ("\n\nAssistant:", ""),
        },
    )
)


register_chat_template(
    ChatTemplate(
        name="chatml",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": ("<|im_start|>system\n", "\n<|im_end|>\n"),
            "user": ("<|im_start|>user\n", "\n<|im_end|>\n"),
            "assistant": ("<|im_start|>assistant\n", "\n<|im_end|>\n"),
        },
        style=ChatTemplateStyle.PLAIN,
    )
)


register_chat_template(
    ChatTemplate(
        name="vicuna_v1.1",
        default_system_prompt=(
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
        ),
        role_prefix_and_suffix={
            "system": ("", " "),
            "user": ("USER:", " "),
            "assistant": ("ASSISTANT:", "</s>"),
        },
        image_token=" <image>\n",
    )
)


register_chat_template(
    ChatTemplate(
        name="llama-2-chat",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": ("<<SYS>>\n", "\n<</SYS>>\n\n"),
            "user": ("[INST] ", " [/INST]"),
            "assistant": ("", " </s><s>"),
        },
        style=ChatTemplateStyle.LLAMA2,
    )
)


@register_chat_template_matching_function
def match_vicuna(model_path: str):
    if "vicuna" in model_path.lower():
        return get_chat_template("vicuna_v1.1")
    if "llava" in model_path.lower():
        return get_chat_template("vicuna_v1.1")


@register_chat_template_matching_function
def match_llama2_chat(model_path: str):
    model_path = model_path.lower()
    if "llama-2" in model_path and "chat" in model_path:
        return get_chat_template("llama-2-chat")
    if (
        "mistral" in model_path or "mixtral" in model_path
    ) and "instruct" in model_path:
        return get_chat_template("llama-2-chat")
    if "codellama" in model_path and "instruct" in model_path:
        return get_chat_template("llama-2-chat")


@register_chat_template_matching_function
def match_chat_ml(model_path: str):
    model_path = model_path.lower()
    if "tinyllama" in model_path:
        return get_chat_template("chatml")
    if "qwen" in model_path and "chat" in model_path:
        return get_chat_template("chatml")


if __name__ == "__main__":
    messages = [
        {"role": "system", "content": None},  # None means default
        # {"role": "system", "content": "You are a helpful, respectful and honest assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi!"},
        {"role": "user", "content": "What can you do?"},
        {"role": "assistant", "content": "I can chat with you."},
    ]

    template = get_chat_template("llama-2-chat")
    print(template.get_prompt(messages))
