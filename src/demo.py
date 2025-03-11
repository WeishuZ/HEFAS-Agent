import json

# JSON 数组字符串
text = '[["产品使用", "登录问题", "验证码错误", "登录时验证码不正确", "为什么我登录时验证码显示错误？"]]'

# 解析 JSON 字符串为数组
array = json.loads(text)

print(array)  # ✅ [['产品使用', '登录问题', '验证码错误', '登录时验证码不正确', '为什么我登录时验证码显示错误？']]
print(type(array))  # ✅ <class 'list'>
