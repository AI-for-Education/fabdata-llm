from os import getenv
import string
import re


# load openai properties
openai_prompt_suffix = str(getenv('OPENAI_PROMPT_SUFFIX'))


BANNED_WORDS = [
  "arse"
  ,"arsehole"
  ,"arses"
  ,"bastard"
  ,"bastards"
  ,"beaver"
  ,"beef curtains"
  ,"bellend"
  ,"bellends"
  ,"bollock"
  ,"bollocks"
  ,"berk"
  ,"berks"
  ,"bint"
  ,"bints"
  ,"bitch"
  ,"bitches"
  ,"bitching"
  ,"blimey"
  ,"blimey o'reilly"
  ,"bloodclaat"
  ,"bloody hell"
  ,"bollocks"
  ,"bugger"
  ,"buggers"
  ,"bugger me"
  ,"bugger off"
  ,"bukkake"
  ,"bullshit"
  ,"cack"
  ,"chav"
  ,"cheese eating surrender monkey"
  ,"choad"
  ,"chuffer"
  ,"clunge"
  ,"clunges"
  ,"cobblers"
  ,"cock"
  ,"cocks"
  ,"cock cheese"
  ,"cock jockey"
  ,"cocksucker"
  ,"cocksuckers"
  ,"cock sucker"
  ,"cock suckers"
  ,"cock-up"
  ,"cockwomble"
  ,"codger"
  ,"cor blimey"
  ,"crap"
  ,"craps"
  ,"crikey"
  ,"cunt"
  ,"cunts"
  ,"daft"
  ,"daft cow"
  ,"damn"
  ,"dick"
  ,"dicks"
  ,"dickhead"
  ,"dickheads"
  ,"dildo"
  ,"dodgy"
  ,"duffer"
  ,"fanny"
  ,"fanyys"
  ,"feck"
  ,"fuck"
  ,"fucks"
  ,"fucking"
  ,"fucker"
  ,"fucktard"
  ,"gob shite"
  ,"goddam"
  ,"gorblimey"
  ,"gordon bennett"
  ,"gormless"
  ,"he's a knob"
  ,"hobknocker"
  ,"jesus christ"
  ,"jizz"
  ,"knobber"
  ,"knobend"
  ,"knobhead"
  ,"ligger"
  ,"mad as a hatter"
  ,"manky"
  ,"minge"
  ,"minges"
  ,"minger"
  ,"minging"
  ,"motherfucker"
  ,"munter"
  ,"muppet"
  ,"naff"
  ,"nitwit"
  ,"nonce"
  ,"numpty"
  ,"nutter"
  ,"off their rocker"
  ,"pillock"
  ,"pish"
  ,"piss off"
  ,"piss-flaps"
  ,"piss"
  ,"pissed"
  ,"pissed off"
  ,"play the five-fingered flute"
  ,"plonker"
  ,"ponce"
  ,"poof"
  ,"pouf"
  ,"poxy"
  ,"prat"
  ,"prick"
  ,"pricks"
  ,"prickteaser"
  ,"punani"
  ,"punny"
  ,"pussy"
  ,"randy"
  ,"rapey"
  ,"rat arsed"
  ,"rotter"
  ,"scrubber"
  ,"shag"
  ,"shags"
  ,"shit"
  ,"shits"
  ,"shite"
  ,"shites"
  ,"shitfaced"
  ,"skank"
  ,"skanks"
  ,"slag"
  ,"slags"
  ,"slapper"
  ,"slappers"
  ,"slut"
  ,"sluts"
  ,"snatch"
  ,"sod-off"
  ,"son of a bitch"
  ,"spunk"
  ,"stick it up your arse!"
  ,"swine"
  ,"taking the piss"
  ,"tits"
  ,"toff"
  ,"tosser"
  ,"trollop"
  ,"tuss"
  ,"twat"
  ,"twats"
  ,"twonk"
  ,"twonks"
  ,"u fukin wanker"
  ,"wally"
  ,"wanker"
  ,"wankers"
  ,"wankstain"
  ,"wazzack"
  ,"whore"
  ,"whores"
]


def contains_banned_words(test_string: str) -> bool:
  has_banned = False
  for word in BANNED_WORDS:
    if re.search(r'\b' + word.upper() + r'\b', test_string.upper()):
        has_banned = True
        break

  return has_banned


def build_prompt_string(input_str: str) -> str:
  final_str = input_str.strip()
  split_str = final_str.split("##")
  if len(split_str) > 1:
    final_str = split_str[1]
  else:
    final_str = split_str[0]

  if len(final_str) > 0:
    last_char = final_str[len(final_str) -1]
    # Create a regex pattern to match all special characters in string
    pattern = r'[' + string.punctuation + ']'
    # Remove special characters from the string
    final_str = re.sub(pattern, '', final_str)
    # Remove multiple spaces
    final_str = re.sub(' {2,}', ' ', final_str)
    # add a question mark back - the regex will remove any existing ones
    if last_char == "?":
      final_str = final_str + '?'
    # append the prompt suffix
    final_str = final_str + str(openai_prompt_suffix)

  return final_str

def clean_prompt_string(prompt: str):
  return prompt.replace(f"{openai_prompt_suffix}", "").strip()