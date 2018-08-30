import inspect

def cheat(code):
    cheat_dict = {
        '0': "This is a cheat for Exercise 0!" 
    }
    if code in cheat_dict:
        print(inspect.cleandoc(cheat_dict[code]))
    else:
        print('Invalid exercise id, Naughty cheater!')

def hint(code):
    hint_dict = {
        '0': "This is a hint for Exercise 0!" 
    }
    if code in hint_dict:
        print(inspect.cleandoc(hint_dict[code]))
    else:
        print('Invalid exercise id. Did you type the id correctly?')

