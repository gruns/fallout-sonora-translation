#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-

'''
Usage:
  translate.py <inputFileOrDir> <outputFileOrDir> --api-key=<apiKey>

Options:
  <inputFileOrDir>      Input .msg file path or directory containing .msg files to translate.
  <outputFileOrDir>     Output .msg file path or directory to write translated file(s) to.
  --api-key=<apiKey>    Your ChatGPT API key.
  -h --help             Show this help message and exit.
'''

import os
import time
import openai
from icecream import ic
from docopt import docopt


def translateText(text, model='gpt-3.5-turbo-16k', temperature=0.8):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                'role': 'system',
                'content': '''You are a translation agent translating Fallout Sonora .msg game files from Russian to English. Fallout Sonora is a Fallout 2 mod written in Russian. You will be provided a file in a specific format that contains Russian, and your task is to translate the Russian into English and replace the Russian with English, in place, maintaining the format of the .msg file.''',
            },
            {
                'role': 'user',
                'content': text,
            },
        ],
        # play with different temperatures
        #   Low temperature (0 to 0.3): More focused, coherent, and conservative outputs.
        #   Medium temperature (0.3 to 0.7): Balanced creativity and coherence.
        #   High temperature (0.7 to 1): Highly creative and diverse, but potentially less coherent.
        temperature=temperature,
        max_tokens=None,
    )

    return response.choices[0].message.content, model, temperature


def main():
    args = docopt(__doc__)

    inputFileOrDir = args['<inputFileOrDir>']
    outputFileOrDir = args['<outputFileOrDir>']
    apiKey = args['--api-key']

    openai.api_key = apiKey

    inputFiles = []
    if inputFileOrDir.endswith('.msg'):
        inputFiles.append(inputFileOrDir)
    else:
        for filename in os.listdir(inputFileOrDir):
            if filename.endswith('.msg'):
                inputFile = os.path.join(inputFileOrDir, filename)
                inputFiles.append(inputFile)
    
    if len(inputFiles) > 1 and outputFileOrDir.endswith('.msg'):
        msg = 'Error: multiple input files but only one output file provided.'
        raise ValueError(msg)

    for i, inputFile in enumerate(inputFiles):
        filename = os.path.basename(inputFile)
        try:
            with open(inputFile, 'r', encoding='windows-1251') as f:
                text = f.read()

            print(f'Translating {filename}... ', end='', flush=True)

            startTime = time.time()
            model = 'gpt-4'
            #model = 'gpt-3.5-turbo-16k'
            temperature = 0
            translatedText, model, temperature = translateText(text, model, temperature)
            endTime = time.time()
            duration = round(endTime - startTime, 3)

            if translatedText:
                if outputFileOrDir.lower().endswith('.msg'):
                    outputFilePath = outputFileOrDir
                else:
                    if outputFileOrDir == '.':
                        outputFileOrDir = os.getcwd()
    
                    name, ext = os.path.splitext(filename)
                    outputFilename = f'{name}-[{model}]-[t={temperature}]{ext.lower()}'
                    outputFilePath = os.path.join(outputFileOrDir, outputFilename)
    
                with open(outputFilePath, 'w', encoding='utf-8') as f:
                    f.write(translatedText)
    
                print(f'to {outputFilePath} in {duration}s.')
        except Exception as e:
            raise
    
        if i > 3:
            break


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
