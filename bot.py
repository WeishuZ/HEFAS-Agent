import discord
from discord.ext import commands
import os
from dotenv import load_dotenv
from agent_api import ChatAgent

# 加载环境变量
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
if not TOKEN:
    raise ValueError("❌ Railway 未正确加载 `DISCORD_TOKEN`，请检查环境变量！")
print(f"✅ 读取到 DISCORD_TOKEN: {TOKEN[:5]}********")

# 设置机器人
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# 初始化聊天代理
agent = ChatAgent()

@bot.event
async def on_ready():
    print(f'{bot.user} 已成功连接到Discord!')

@bot.event
async def on_message(message):
    # 如果消息来自机器人本身，则忽略
    if message.author == bot.user:
        return

    # 处理用户消息
    if message.content:
        try:
            # 使用聊天功能
            response = agent.chat(message.content, history=None)
            
            # 发送响应
            await message.channel.send(response)
        except Exception as e:
            await message.channel.send("抱歉，处理消息时出现错误。")
            print(f"错误: {str(e)}")

    # 允许命令处理
    await bot.process_commands(message)

# 运行机器人
bot.run(TOKEN) 