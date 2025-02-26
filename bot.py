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
    if message.author == bot.user:
        return  

    if bot.user.mentioned_in(message):  
        try:
            response = agent.chat(message.content, history=None)
            await message.channel.send(response)
        except Exception as e:
            await message.channel.send("Sorry, I have some issue while dealing the message。")
            print(f"Error: {str(e)}")

    await bot.process_commands(message)  # 允许命令继续执行


# 运行机器人
bot.run(TOKEN) 