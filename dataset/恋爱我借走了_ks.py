def get_all_files_in_directory(directory, ext=''):
    import os
    import re
    custom_sort_key_re = re.compile('([0-9]+)')

    def custom_sort_key(s):
        # 将字符串中的数字部分转换为整数，然后进行排序
        return [int(x) if x.isdigit() else x for x in custom_sort_key_re.split(s)]

    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(ext):
                file_path = os.path.join(root, file)
                all_files.append(file_path)
    return sorted(all_files, key=custom_sort_key)


def clearT():
    import unicodedata
    from opencc import OpenCC

    def full2half(input_str):
        return ''.join([unicodedata.normalize('NFKC', char) for char in input_str])

    cc = OpenCC('t2s')  # 't2s'表示繁体转简体

    def _clearT(s):
        s = cc.convert(full2half(s))
        return s.strip().strip(r'\n').replace('\n', '\\n')

    return _clearT


clearT = clearT()


def startsWithAny(s: str, keys):
    for x in keys:
        if s.startswith(x):
            return x
    else:
        return False


def startsWithAlnum(s: str, _chrs):
    retn = ''
    for char in s:
        if (char.isalnum()) or (char in _chrs):
            retn += char
        else:
            break
    return retn


def startsWithCmd(s: str, _chrs=None):
    if _chrs is None:
        _chrs = {'_', '@'}
    cmd = startsWithAlnum(s, _chrs)
    if cmd:
        return s.startswith(cmd + ' ')
    else:
        return False


import re

reg_removeWait = re.compile(r'\[.+?]')


def removeWait(_line):
    _tmp = reg_removeWait.sub('', _line)
    if _tmp:
        return _tmp
    else:
        return _line


def getCmdArgs(_cmd):
    _args = ['']
    _i = -1
    _inQuota = ''
    while _i < len(_cmd) - 1:
        _i += 1
        char = _cmd[_i]
        if _inQuota:
            _args[-1] += char
            if char == _inQuota:
                _inQuota = ''
            elif char == '\\':
                _i += 1
                char = _cmd[_i]
                _args[-1] += char
            else:
                pass
        else:
            if char == ' ':
                _args.append('')
                while _i < len(_cmd) - 1:
                    _i += 1
                    char = _cmd[_i]
                    if char != ' ':
                        _i -= 1
                        break
                continue
            elif char in {'"', "'"}:
                _inQuota = char
            else:
                pass
            _args[-1] += char
            continue
    return _args


def args2map(_args):
    retn = {}
    for _i in range(1, len(_args)):
        _arg = _args[_i]
        if '=' not in _arg:
            if '＝' in _arg:
                _arg = _arg.replace('＝', '=')
            else:
                continue
        _idx = _arg.index('=')
        _k = _arg[:_idx]
        if '"' in _k or "'" in _k:
            continue
        _v = _arg[_idx + 1:]
        if _v.startswith('"') or _v.startswith("'"):
            _v = _v[1:len(_v) - 1]
        retn[_k] = _v
    return retn


# =================

a = get_all_files_in_directory(r'E:\tmp\恋爱我借走了\ks', ext='.ks')
b = r'D:\datasets\tmp'

# =================

sc = {}

_n = {
    "咲希": "咲希",
    "桃子": "桃子",
    "绘未": "绘未",
    "幸": "幸",
    "八纯": "八纯",
    "绘未妈妈": "绘未妈妈",
    "椿": "椿",
    "千夏": "千夏",
    "梦乃": "梦乃",
    "吾郎": "吾郎",
    "小夏": "小夏",
    "？？？": "?",
    "椿母": "椿母",
    "爱季": "爱季",
    "複数": "复数",
    "月": "月",
    "Ｍａｓｔｅｒ": "Master",
    "主人公": "主人公",
    "千夏小夏母亲": "千夏小夏母亲",
    "走散的长颈鹿": "走散的长颈鹿",
    "海豹": "海豹",
    "向导": "向导",
    "新海家的亲戚": "新海家的亲戚",
    "卡拉ＯＫ店员": "卡拉OK店员",
    "天真的男孩": "天真的男孩",
    "见多识广的女孩": "见多识广的女孩",
    "拉面摊大叔": "拉面摊大叔",
    "拉面摊的大叔": "拉面摊的大叔",
    "超级不感兴趣的长颈鹿": "超级不感兴趣的长颈鹿",
    "月的班主任": "月的班主任",
    "急性子的斋藤": "急性子的斋藤",
    "懦弱的川下": "懦弱的川下",
    "粗野的豪田": "粗野的豪田",
    "做主角的幸": "做主角的幸"
}

# =================
for path in a:
    name = path[path.rindex('\\'):]
    name = '0'
    if name not in sc:
        sc[name] = []
        print(name)
    # =================

    with open(path, 'r', encoding='utf-16-le') as f:
        data = list(filter(lambda x: x and (not x.startswith(';')),
                           (x.rstrip() for x in f.readlines())))

    # =================
    w_i = -1
    while w_i < len(data) - 1:
        w_i += 1
        line: str = data[w_i]
        # =================
        if not line.endswith('[c]'):
            continue
        # =================
        if line.startswith('[地'):
            n = '旁白'
        else:
            if '[>> tx="『"]' in line:
                line = line.replace('[>> tx="『"]', '[>>]')
                line = line.replace('[<< tx="』"]', '[<<]')
            assert line.endswith('[<<][c]') and '[>>]' in line
            if line.startswith('[>>]'):
                n: str = data[w_i-1]
            else:
                print(line)
                n = line[:line.index('[>>]')]
                line = line[len(n):]
                print(n, line)

            if n.startswith('[ネーム表示'):
                n = n[n.index('【')+1:n.rindex('】')]
            else:
                if ' ' in n:
                    n = n[1:n.index(' ')]
                else:
                    n = n.strip('[]')

            if n in _n:
                n = _n[n]
            else:
                _n[n] = clearT(n)
                print(line)

        line = removeWait(line)
        d = clearT(line)
        if d:
            sc[name].append(n + '：' + d)

# =================

for k, v in sc.items():
    if v:
        with open(b + f'\\{k}.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(v))

# =================
import json
tmp = json.dumps(_n, ensure_ascii=False, indent=4)