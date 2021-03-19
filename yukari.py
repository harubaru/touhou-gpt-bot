import argparse
import time
from pastebin import PastebinAPI
from discord import Status
from discord import Game
from discord.ext.commands import Bot
from discord import Embed
from discord import Colour

from story import *
from memory import *

start_time = time.time()

parser = argparse.ArgumentParser(description='Input argument parser.')

parser.add_argument('--token', type=str, help='Bot token. Do not share this!')
parser.add_argument('--prefix', type=str, help='Prefix for interacting with the bot.', default='y!')
parser.add_argument('--context', type=str, help='Context string that is prefixed in every Act statement. It is filled with information that the AI should remember. This information is needed if you wish to inquire the AI about topics that haven\'nt been trained into it\'s model.',
        default='You are a man from the Outside World who goes by the name \"Anon\". You are relaxing one day when Yukari decided to materialize in your room. Yukari is a woman who sleeps all day and lives for the enjoyment of life. She is well-acquainted with everyone in Gensokyo and knows a lot of people. She is a very wise woman and is an expert at mathematics. Yukari\'s current intentions are solely to have a friendly conversation with you while you are relaxing in your room. You engage in a conversation with Yukari and she is the person that you are talking to.\nYou say, \"Hello, Yukari!\".\nYukari says, \"Hello, Anon.\".')

parser.add_argument('--model_dir', type=str, help='path of model folder')
parser.add_argument('--custom_model', type=str, help='path to custom model')
parser.add_argument('--nucleus', help='flag to turn on/off nucleus sampling', action='store_true')
parser.add_argument('--top_p', type=float, help='cut off probablity for nucleus sampling', default=0.9)
parser.add_argument('--top_k', type=int, help='cut off ranking for top K sampling', default=20)
parser.add_argument('--temperature', type=float, help='temperature in text generation. Higher temperature creates more randomness in the results.', default=0.8)
parser.add_argument('--batch_size', type=int, help='batch size', default=1)
parser.add_argument('--output_length', type=int, help='length of output sequence (number of tokens)', default=500)
parser.add_argument('--past_length', type=int, help='amount of memorized inputs and responses', default=16)
parser.add_argument('--mem_path', type=str, help='path to memories json file', default='yukarimemory.json')
parser.add_argument('--gpu_index', type=int, help='which GPU to inference the AI on', default=None)
parser.add_argument('--gpu_max_mem', type=int, help='sets max GPU VRAM usage in megabytes', default=4096)
parser.add_argument('--timeout', type=int, help='Max time in seconds for an act job to run for', default=120)

# pastebin stuff
parser.add_argument('--pastebin_key', type=str, help='Pastebin dev key')

args = parser.parse_args()

client = Bot(command_prefix=args.prefix)
paste  = PastebinAPI()

embed_colour = Colour.from_rgb(178, 0, 255)

def pastebin_upload(content):
        return paste.paste(api_dev_key=args.pastebin_key, api_paste_code=content, api_user_key=None, paste_name=None, paste_format=None, paste_private='unlisted', paste_expire_date='N')

def trunc(num, digits):
        sp = str(num).split('.')
        return '.'.join([sp[0], sp[1][:digits]])

def log(com, logstr):
        timestamp = time.time() - start_time
        print('[' + trunc(timestamp, 4) + '] ' + com + ': ' + logstr)

def actjob(message, author):
        if message != '':
                log('ai  ', 'Processing Act job -- [' + message + ']')
                message = message + '\n'
        else:
                log('ai  ', 'Processing Redo job')
        
        output = run_model(args, message)
        logged_output = output.replace('\n', '')

        log('ai  ', 'Generated result -- [' + logged_output + ']')

        embed = Embed(description="```" + output + '```\n\n**Response generated for:** ``' + author + '``', colour=embed_colour)

        return embed

@client.event
async def on_ready():
        log('init', 'Connected to Discord servers.')
        game = Game('with Ran\'s tails~')
        await client.change_presence(status=Status.online, activity=game)

@client.command(name='reset',
                description='Resets AI execution. This is useful if you want to start the AI over from the beginning.',
                brief='Resets the AI model.',
                aliases=['r'],
                pass_context=True)
async def resetcmd(context):
        log('ai  ', 'Restarting AI inferencer...')
        args.input_stack = []
        log('ai  ', 'Done!~')
        await context.message.channel.send(embed=Embed(title="Done~!", colour=embed_colour))

@client.command(name='temp',
                description='Sets the temperature for the AI. Higher number == More random results.',
                brief='Sets the temperature for the AI.',
                pass_context=True)
async def tempcmd(context):
        log('ai  ', 'Changing temperature to ' + context.message.content[7:])
        args.temperature = float(context.message.content[7:])

@client.command(name='top_k',
                description='Cut off ranking for top K sampling. Higher number == More creative results',
                brief='Cut off ranking for top K sampling.',
                pass_context=True)
async def top_kcmd(context):
        log('ai  ', 'Changing Top_K to ' + context.message.content[8:])
        args.top_k = float(context.message.content[8:])

@client.command(name='top_p',
                description='Adjust the summed probabilities of what tokens should considered to be generated.',
                brief='Use this with Nucleus sampling to get rid of neural degeneration.',
                pass_context=True)
async def top_pcmd(context):
        log('ai  ', 'Changing Top_P to ' + context.message.content[8:])
        args.top_p = float(context.message.content[8:])

@client.command(name='nucleus',
                description='Toggle nucleus sampling',
                brief='Toggle this to solve repetition.',
                pass_context=True)
async def nucleuscmd(context):
        args.nucleus = not args.nucleus
        if args.nucleus == True:
                log('ai  ', 'Enabled Nucleus Sampling')
                await context.message.channel.send(embed=Embed(title="Nucleus sampling enabled.", colour=embed_colour))
        else:
                log('ai  ', 'Disabled Nucleus Sampling')
                await context.message.channel.send(embed=Embed(title="Nucleus sampling disabled.", colour=embed_colour))

@client.command(name='memlength',
                description='Length of how many actions it will remember',
                brief='Length of how many actions it will remember',
                pass_context=True)
async def lengthcmd(context):
        log('ai  ', 'Changing output_length to ' + context.message.content[12:])
        args.past_length = int(context.message.content[12:])

@client.command(name='outlength',
                description='Change length of generated output',
                brief='Output length',
                pass_context=True)
async def outlengthcmd(context):
        log('ai  ', 'Changing output_length to ' + context.message.content[12:])
        args.output_length = int(context.message.content[12:])

@client.command(name='raw',
                description='Feed raw input to Yukari!',
                brief='Feed raw input to Yukari!',
                pass_context=True)
async def rawcmd(context):
        message = context.message.content[6:]
        
        await context.message.channel.send(embed=actjob(message, context.message.author.name + '#' + context.message.author.discriminator))

@client.command(name='say',
                description='Say something to Yukari!',
                brief='Say something to Yukari!',
                aliases=['s'],
                pass_context=True)
async def saycmd(context):
        message = "You say, \"" + context.message.content[6:] + "\""

        await context.message.channel.send(embed=actjob(message, context.message.author.name + '#' + context.message.author.discriminator))

@client.command(name='do',
                decsription='Do something to Yukari!',
                brief='Do something to Yukari!',
                aliases=['d'],
                pass_context=True)
async def docmd(context):
        message = context.message.content[5:]
        message = "You " + message[0].lower() + message[1:]
        if message[-1:] != '.':
                message = message + '.'

        await context.message.channel.send(embed=actjob(message, context.message.author.name + '#' + context.message.author.discriminator))

@client.command(name='redo',
                description='Redo an action',
                brief='Redo an action',
                pass_context=True)
async def redocmd(context):
        args.input_stack.pop()
        
        await context.message.channel.send(embed=actjob('', context.message.author.name + '#' + context.message.author.discriminator))

@client.command(name='undo',
                description='Undo an action',
                brief='Undo an action',
                pass_context=True)
async def undocmd(context):
        args.input_stack.pop()
        log('ai  ', 'Undoed an action')
        await context.message.channel.send(embed=Embed(title='Reverted!', colour=embed_colour))

@client.command(name='print',
                description='Print conversation',
                brief='Print the conversation',
                pass_context=True)
async def printcmd(context):
        args.pastebin_key = None
        if args.pastebin_key == None:
                await context.message.channel.send(''.join(args.input_stack))
        else:
                await context.message.channel.send(pastebin_upload(''.join(args.input_stack)))

@client.command(name='ctx',
                description='Set new context for AI',
                brief='Set new context',
                pass_context=True)
async def ctxcmd(context):
        newcontext = context.message.content[6:]
        args.context = newcontext
        log('ai  ', 'Set context to -- [' + newcontext + ']')
        await context.message.channel.send(embed=Embed(title='Set context!', colour=embed_colour))

@client.command(name='forget',
                description='Make Yukari forget something!',
                brief='Make Yukari forget something',
                pass_context=True)
async def forgetcmd(context, key):
        mem_delete(key)
        mem_save(args.mem_path)
        log('mem ', 'Forgetting key [' + key + ']')

@client.command(name='remember',
                description='Make Yukari permanently remember something!',
                brief='Make Yukari remember something',
                pass_context=True)
async def remembercmd(context, key, description):
        mem_encode(key, description)
        mem_save(args.mem_path)
        log('mem ', 'Remembering key [' + key + '] as {' + description + '}')

@client.command(name='memories',
                description='List memories',
                brief='List saved memories',
                pass_context=True)
async def memoriescmd(context):
        message = 'Saved Memories:\n'
        for key, desc, in mem_dict().items():
                message = message + '**' + key + '** : ``' + desc + '``\n'
        
        await context.message.channel.send(message)

@client.command(name='params',
                description='List AI parameters',
                brief='List params',
                pass_context=True)
async def paramscmd(context):
        description = "__Processing Parameters__\n\nTop_K = ``" + str(args.top_k) + "``\nTop_P = ``" + str(args.top_p) + "``\nTemp = ``" + str(args.temperature) + "``\nRep_P = ``1.4``\nNucleus Sampling = ``"
        if args.nucleus == True:
                description = description + "Enabled``"
        else:
                description = description + "Disabled``"

        description = description + "\n\n__Memory Info__\n\nKnowledge Entries = ``" + str(len(mem_dict())) + "``\nRemembered Actions = ``" + str(len(args.input_stack)) + "``"
        await context.message.channel.send(embed=Embed(title='AI Parameters', description=description, colour=embed_colour))

def main():
        if not args.token:
                print('token must be provided')
                exit()

        log('init', 'Server started at ' + time.strftime("%Y-%m-%d %H:%M"))

        if mem_load(args.mem_path):
                log('init', 'Loaded memories')
        else:
                log('init', 'Memory file not found, creating a new one')

        log('init', 'Initializing model inferencer')
        init_connection(args)
        client.run(args.token)

if __name__ == '__main__':
        main()
