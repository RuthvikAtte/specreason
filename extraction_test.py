def extract_boxed(text):
    start_idx = text.find("\\boxed{")
    if start_idx == -1:
        return None
    
    start_idx += len("\\boxed{")
    brace_count = 1
    for i in range(start_idx, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[start_idx:i]
    return None

print(extract_boxed(r"Some text \boxed{\frac{\pi}{2}} more text"))
print(extract_boxed(r"Some text \boxed{123}"))
print(extract_boxed(r"No boxed here"))
