# Text based long term memory.

import os
import json
import re

memories = {}

def mem_encode(keyword, inputstr):
        memories[keyword.lower()] = inputstr

def mem_retrieve(keyword):
        if keyword.lower() in memories:
                return memories[keyword.lower()]
        
        return ''

def mem_delete(keyword):
        if keyword.lower() in memories:
                del memories[keyword.lower()]

# Compiles the context for memories in a string
def mem_compile(inputstr):
        compiled = ''
        splitstr = inputstr.split(' ')
        for i in range(len(splitstr)):
                splitstr[i] = ''.join(filter(str.isalnum, splitstr[i])).lower()
                retrievedstr = mem_retrieve(splitstr[i])
                if retrievedstr != '':
                        compiled = compiled + retrievedstr + '\n'
        
        return compiled


def mem_save(filepath):
        fp = open(filepath, "w")
        json.dump(memories, fp)

def mem_load(filepath):
        global memories
        try:
                fp = open(filepath, "r")
                if fp:
                        memories = json.load(fp)
                
                return True
        except OSError as e:
                print('Could not find ' + filepath)
                return False

def mem_dict():
        return memories
