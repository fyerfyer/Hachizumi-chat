"""数据构建层全局配置。"""
from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv() -> None:
    """从项目根目录 .env 文件加载环境变量（不依赖 python-dotenv）。"""
    root = Path(__file__).resolve().parent.parent.parent
    env_path = root / ".env"
    if not env_path.exists():
        return
    try:
        with env_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        pass


_load_dotenv()

# ---------------- 路径 ----------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LOG_DIR = PROJECT_ROOT / "logs"

RAW_GZ = PROJECT_ROOT / "dataset" / "0.txt.gz"
SCENES_PATH = PROCESSED_DIR / "scenes.jsonl"
PARSE_STATS_PATH = PROCESSED_DIR / "stats.json"
CHARACTER_CARD_PATH = DATA_DIR / "character_card_hasumi.json"
SYSTEM_PROMPT_PATH = DATA_DIR / "system_prompt.txt"
SFT_TRAIN_PATH = DATA_DIR / "sft_train.jsonl"
SFT_TEST_PATH = DATA_DIR / "sft_test.jsonl"
DPO_TRAIN_PATH = DATA_DIR / "dpo_train.jsonl"

# 自动创建目录
for _d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------- 角色与解析 ----------------
NARRATOR_NAME = "旁白"
TARGET_ROLE = "八纯"
PROTAGONIST = "幸"
UNKNOWN = "未知"

# 已知角色名 -> 规范名
SPEAKER_ALIASES = {
    # 主要角色
    "咲希": "咲希",
    "桃子": "桃子",
    "绘未": "绘未",
    "幸": "幸",
    "八纯": "八纯",
    "主人公": "主人公",
    "椿": "椿",
    "千夏": "千夏",
    # 次要角色 / 别名
    "绘未妈妈": "绘未妈妈",
    "梦乃": "梦乃",
    "吾郎": "吾郎",
    "小夏": "小夏",
    "月": "月",
    "Ｍａｓｔｅｒ": "Master",
    "Master": "Master",
    "爱季": "爱季",
    "複数": "复数",
    "复数": "复数",
    "千夏小夏母亲": "千夏小夏母亲",
    "走散的长颈鹿": "走散的长颈鹿",
    "海豹": "海豹",
    "向导": "向导",
    "新海家的亲戚": "新海家的亲戚",
    "卡拉ＯＫ店员": "卡拉OK店员",
    "卡拉OK店员": "卡拉OK店员",
    "天真的男孩": "天真的男孩",
    "见多识广的女孩": "见多识广的女孩",
    "拉面摊大叔": "拉面摊大叔",
    "拉面摊的大叔": "拉面摊大叔",
    "超级不感兴趣的长颈鹿": "超级不感兴趣的长颈鹿",
    "月的班主任": "月的班主任",
    "急性子的斋藤": "急性子的斋藤",
    "懦弱的川下": "懦弱的川下",
    "粗野的豪田": "粗野的豪田",
    "做主角的幸": "做主角的幸",
    "？": "?",
    "？？？": "?",
    "?": "?",
}

# 场景切分参数
SCENE_BOUNDARY_NARRATOR_STREAK = 3
MAX_GAP_WITHOUT_TARGET = 20
MAX_SCENE_LINES = 120
MIN_SCENE_LINES = 3
LOCATION_KEYWORDS = [
    "教室", "泳池", "海边", "家庭餐厅", "餐厅", "校舍", "走廊", "操场",
    "车站", "图书馆", "社团", "房间", "客厅", "厨房", "公园", "商业街",
    "教室", "游泳馆", "游泳池", "天台", "玄关", "卧室", "游戏中心", "便利店",
]

# ---------------- SFT 参数 ----------------
SFT_WINDOW_TURNS_MIN = 4
SFT_WINDOW_TURNS_MAX = 8
SFT_CONTEXT_TURNS_MIN = 1
SFT_CONTEXT_TURNS_MAX = 2
SFT_MIN_ASSISTANT_LEN = 5
SFT_DEDUP_SIM_THRESHOLD = 0.80
SFT_TRAIN_TEST_SPLIT = 0.90
SFT_RANDOM_SEED = 42

# ---------------- DPO 参数 ----------------
DPO_DEFAULT_PAIRS = 1500
DPO_DEFAULT_SYNTH_SFT = 2000
DPO_SEED_QUESTIONS_PER_TYPE = 50
DPO_PARAPHRASE_PER_QUESTION = 20

# ---------------- API 参数 ----------------
# 新版通用 LLM 配置（推荐）
LLM_API_KEY = os.environ.get("LLM_API_KEY", os.environ.get("KIMI_API_KEY", ""))
LLM_BASE_URL = os.environ.get(
    "LLM_BASE_URL", os.environ.get("KIMI_BASE_URL", "https://api.deepseek.com/v1")
)
LLM_MODEL = os.environ.get(
    "LLM_MODEL", os.environ.get("KIMI_MODEL", "deepseek-chat")
)
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "3"))
LLM_RPM = int(os.environ.get("LLM_RPM", "60"))
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "120"))

# 保留旧版 Kimi 别名，兼容历史代码
KIMI_API_KEY = LLM_API_KEY
KIMI_BASE_URL = LLM_BASE_URL
KIMI_MODEL = LLM_MODEL
KIMI_MAX_RETRIES = LLM_MAX_RETRIES
KIMI_RPM = LLM_RPM
KIMI_TIMEOUT = LLM_TIMEOUT

# ---------------- Prompt 模板 ----------------
SYSTEM_PROMPT_TEMPLATE = (
    "你是天满八纯，绘未的青梅竹马兼同班同学。你成绩优秀、性格友好又有点调皮，"
    "说话活泼直接，偶尔会小小地捉弄亲近的人。请用符合你性格的语气回答。\n"
    "当前场景：{scene_description}"
)

SYNTH_SFT_PROMPT_TEMPLATE = """请扮演游戏《恋爱，我借走了》中的角色天满八纯。
角色设定：
- 绘未是你的青梅竹马兼同班同学。
- 幸是你的同班同学，与「租赁恋人」的委托有关，不是男朋友。
- 你性格友好调皮、好奇心旺盛，偶尔有点小恶魔/腹黑。
- 常用语气：活泼、轻快、偶尔带捉弄，可能会说「诶——」「真是的」「没办法呢」等。
- 不要编造未在游戏中发生的剧情，不要直接复述原游戏剧本。

请围绕「{theme}」生成 {batch_size} 段相互独立的 {min_turns} 到 {max_turns} 轮日常对话。
每段对话是一个 JSON 数组；整体输出一个 JSON 对象，字段为 "conversations"，值是这些数组的列表。
数组元素格式：{{"role": "user"|"assistant", "content": "..."}}
只输出 JSON，不要任何解释。
"""

DPO_QUESTION_GENERATION_PROMPT_TEMPLATE = """基于以下天满八纯角色设定，生成 {n} 个关于她的自然提问。问题应覆盖关系、剧情、口癖、日常等角度。
角色设定：
{character_card_text}

注意：
- 问题要像普通玩家/朋友会问的日常问题；
- 不要涉及 R18 内容；
- 输出 JSON 数组：[{{"question": "...", "type": "relationship|plot|speech|daily"}}]
"""

DPO_PARAPHRASE_PROMPT_TEMPLATE = """把下面的问题改写成 {n} 种不同的日常问法，保持原意但措辞自然、口语化。
原问题：{question}
只输出 JSON 数组，元素为字符串。
"""

DPO_BATCH_PAIR_PROMPT_TEMPLATE = """你是天满八纯，请针对以下 {n} 个问题，分别生成 "chosen"（严格按八纯角色设定回答）和 "rejected"（普通 AI 助手平淡回答，可含事实偏差）。

问题列表：
{questions}

角色设定：
- 绘未是你的青梅竹马兼同班同学。
- 幸是你的同班同学，与「租赁恋人」的委托有关，不是男朋友。
- 你性格友好调皮、偶尔小恶魔，语气活泼轻快。
- 不要编造未在游戏中发生的剧情，不要使用现代网络流行语。

只输出 JSON 数组，每个元素：{{"prompt": "原问题", "chosen": "...", "rejected": "..."}}
不要任何解释。
"""
