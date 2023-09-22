#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-

'''
Usage:
  translate.py <inputFileOrDir> <outputFileOrDir> --api-key=<apiKey> [-t <temperature] [-m <model>] [-j <numJobs>] [-e <encoding>]

Options:
  <inputFileOrDir>      Input .msg file path or directory containing .msg files to translate.
  <outputFileOrDir>     Output .msg file path or directory to write translated file(s) to.
  -e <encoding>         The text encoding of the input file. E.g. windows-1251 for Russian. [default: windows-1251]
  -t <temperature>      LLM temperature, from 0..2. 0 less random, 2 most random. [default: 1.5]
  -m <model>            The OpenAI model to use. [default: gpt-3.5-turbo-16k]
  -j <numJobs>          Number of parallel jobs to run. [default: 4]
  --api-key=<apiKey>    Your OpenAI API key.
  -h --help             Show this help message and exit.
'''

import os
import re
import sys
import time
import multiprocessing
import concurrent.futures
from os.path import basename

import openai
import tiktoken
from icecream import ic
from docopt import docopt


#CONCAT_DELIMITER = '--- [translate.py] ---\n'
CONCAT_DELIMITER = '\n\n\n'
PROMPT = '''You are a translation agent translating Fallout Sonora .msg game files from Russian to English. Fallout Sonora is a Fallout 2 mod written in Russian. You will be provided a file in a specific format that contains Russian, and your task is to translate the Russian into English and replace the Russian with English, in place, maintaining the format of the .msg file. The output file should have the same number of lines as the input file. Don't stop translating until all lines have been translated and the output has the same number of lines as the input.'''
#PROMPT = '''You are a translation agent translating lines of dialog from Fallout Sonora, a Russian Fallout 2 mod, from Russian to English. You will be provided with lines of dialog, one line of dialog per line, and your task is to translate each line, line by line, from Russian to English. Translate each line independently. Do not include any Russian in the output, only out the translated English lines.'''


# See https://openai.com/pricing.
MODELS = {
    #  (contextWindowSize, costPerInputToken, costPerOutputToken)
    'gpt-4': (2**13, 0.06 / 1000, 0.12 / 1000),
    'gpt-3.5-turbo-16k': (2**14, 0.003 / 1000, 0.004 / 1000)
}


#You are a translation agent translating Fallout Sonora .msg game files from Russian to English. Fallout Sonora is a Fallout 2 mod written in Russian. You will be provided multiple .msg games files concatenated, back to back, delimited by the delimiter "--- [translate.py] ---". Each file is in a specific format that contains Russian, and your task is to translate the Russian in these files to English and replace the Russian with its translated English, in place, maintaining the format of each .msg file and all the delimiters between .msg files.

# You are a translation agent translating Fallout Sonora .msg game files from Polish to English. Fallout Sonora is a Fallout 2 mod written in Polish. You will be provided a file in a specific format that contains Polish, and your task is to translate the Polish into English and replace the Polish with English, in place, maintaining the format of the .msg file.

# You are a translation agent translating Fallout Sonora .msg game files to English. Fallout Sonora is a Russian Fallout 2 mod. You will be provided two files: one version in Russian and another version in Polish. The Polish version was translated from the Russian version. Your task is to translate the non-English sentences in the Russian file and the Polish file into English, replacing the non-English sentences with the according English, in place, maintaining the format of the .msg file. The Russian file will be provided first, followed by the Polish file. The Polish version should be used a helpful reference to produce a better English translation. Only one version of the final file, in English, should be output.


def verifyMsgFilesMatch(msgFile1, encoding1, msgFile2, encoding2):
    with open(msgFile1, 'r', encoding=encoding1) as f:
        lines1 = f.read().strip().splitlines()
    with open(msgFile2, 'r', encoding=encoding2) as f:
        lines2 = f.read().strip().splitlines()

    # verify they have same number of lines
    if len(lines1) != len(lines2):
        ic('diff number of lines:', len(lines1), len(lines2))
        return False

    for lineno, (line1, line2) in enumerate(zip(lines1, lines2)):
        if line1.startswith('{') or line2.startswith('{'):
            dialogueLineNumber1 = line1.split('}', 1)[0].lstrip('{')
            dialogueLineNumber2 = line2.split('}', 1)[0].lstrip('{')

            # both lines start with the same number, eg {201}{... and {201}{...
            if dialogueLineNumber1 != dialogueLineNumber2:
                print(f'dialogueLineNumber mismatch on line {lineno+1}!')
                return False
            elif not line1.endswith('}') or not line2.endswith('}'):
                print(f'malformed line on {lineno+1}!')
                return False

    return True

    #pattern = r'^{.*}$'
    #for line1, line2 in zip(lines1, lines2):
    #    # if a line starts with '{', it must also end with a '}'. this
    #    # catches llm hallucinations where the llm just goes bananas and
    #    # generates with garbage. this is more likely to happen at
    #    # higher temperatures
    #    if ((line1.startswith('{') and not line2.startswith('{')) or
    #         (line1.endswith('}') and not line2.endswith('}'))):
    #        ic('invalid line format')
    #        return False

    return True


def extractMsgFileText(s):
    ret = []

    for line in s.splitlines():
        line = line.strip()  # Remove leading/trailing whitespace
        lastOpenBrace = line.rfind('{')
        lastCloseBrace = line.rfind('}')
            
        if (lastOpenBrace != -1 and lastCloseBrace != -1
            and lastOpenBrace < lastCloseBrace):
            extractedContent = line[lastOpenBrace+1 : lastCloseBrace]
            ret.append(extractedContent)

    return '\n'.join(ret)


def concatenateLinesOfMsgFiles(inputFiles, encoding, delimiter=None):
    concatenated = ''

    for filePath in inputFiles:
        with open(filePath, 'r', encoding=encoding) as f:
            textOnly = extractMsgFileText(f.read())
            concatenated += delimiter + textOnly + '\n'

    return concatenated


def concatenateFiles(inputFiles, encoding, delimiter=None):
    concatenated = ''

    for filePath in inputFiles:
        with open(filePath, 'r', encoding=encoding) as f:
            concatenated += delimiter + f.read() + '\n'

    return concatenated


def countTokens(s, model):
    encoding = tiktoken.encoding_for_model(model)
    numTokens = len(encoding.encode(s))
    return numTokens


def packInputFilesForContextWindowSize(inputFiles, encoding, model, maxTokensPerGroup):
    groups = []
    currentGroup = []
    currentNumTokens = countTokens(PROMPT, model)

    for inputFile in inputFiles:
        with open(inputFile, 'r', encoding=encoding) as f:
            text = f.read()
            numTokens = countTokens(text, model)

        if currentNumTokens + numTokens <= maxTokensPerGroup:
            currentGroup.append(inputFile)
            currentNumTokens += numTokens
        else:
            groups.append(currentGroup)
            currentGroup = [inputFile]
            currentNumTokens = countTokens(PROMPT, model) + numTokens

    if currentGroup:
        groups.append(currentGroup)

    packed = [
        #(group, concatenateFiles(group, encoding, CONCAT_DELIMITER).strip())
        (group, concatenateLinesOfMsgFiles(
            group, encoding, CONCAT_DELIMITER).strip())
        for group in groups]

    return packed


def calculateOpenAIApiCost(response, model):
    if model not in MODELS:
        return None

    _, costPerInputToken, costPerOutputToken = MODELS[model]
    numInputTokens = response.usage.prompt_tokens
    numOutputTokens = response.usage.completion_tokens

    inputCost = numInputTokens * costPerInputToken
    outputCost = numOutputTokens * costPerOutputToken

    totalCost = inputCost + outputCost

    return totalCost


def translateText(text, model='gpt-3.5-turbo-16k', temperature=1.5):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                'role': 'system',
                'content': PROMPT,
            },
            {
                'role': 'user',
                'content': text,
            },
        ],
        temperature=temperature,
        max_tokens=None,
    )

    cost = calculateOpenAIApiCost(response, model)

    return response.choices[0].message.content, cost


def translateConcatenatedMsgFiles(
        inputFiles, delimitedMsgFiles, outputFileOrDir, model, temperature):
    filenames = [basename(inputFile) for inputFile in inputFiles]
    print(f'Translating {", ".join(filenames)}... ', end='', flush=True)

    print(delimitedMsgFiles)

    startTime = time.time()
    translatedText, cost = translateText(delimitedMsgFiles, model, temperature)
    endTime = time.time()
    duration = round(endTime - startTime, 3)
    
    print(translatedText)


def translateMsgFile(inputFile, inputEncoding, outputFileOrDir, model, temperature):
    filename = basename(inputFile)

    with open(inputFile, 'r', encoding=inputEncoding) as f:
        raw = f.read()

    print(f'Translating {filename}... ', end='', flush=True)

    startTime = time.time()
    #translatedText, cost = translateText(extractMsgFileText(raw), model, temperature)
    translatedText, cost = translateText(raw, model, temperature)
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

        print(f'to {outputFilePath} in {duration}s for ${round(cost, 3)}.')
    else:
        print(f'FAILED in {duration}s.')

    return outputFilePath


def translateAndVerifyMsgFile(
        inputFile, inputEncoding, outputFileOrDir, model, temperature):
    outputFilePath = translateMsgFile(
        inputFile, inputEncoding, outputFileOrDir, model, temperature)
    matching = verifyMsgFilesMatch(
        inputFile, inputEncoding, outputFilePath, 'utf-8')

    if matching:
        print(f'{basename(inputFile)} matches {basename(outputFilePath)}')
    else:
        print(f'{basename(inputFile)} DOES NOT MATCH {basename(outputFilePath)}')


def main():
    args = docopt(__doc__)

    inputFileOrDir = args['<inputFileOrDir>']
    outputFileOrDir = args['<outputFileOrDir>']
    openai.api_key = args['--api-key']
    temperature = float(args['-t'])
    model = args['-m']
    numJobs = int(args['-j'])
    inputEncoding = args['-e']

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

    inputFiles = inputFiles[0:6] # DEBUG

    #ctxWindowSize, _, _ = MODELS[model]
    #ctxWindowSize = ctxWindowSize * 0.95  # leave some extra space, just in case
    #packed = packInputFilesForContextWindowSize(
    #    inputFiles, inputEncoding, model, ctxWindowSize)
    #
    #for (group, delimitedMsgFiles) in packed:
    #    ic(group)
    #    translateConcatenatedMsgFiles(
    #        group, delimitedMsgFiles, outputFileOrDir, model, temperature)
    #    break
    #
    #concatenated = concatenateFiles(
    #    inputFiles, 'windows-1251', CONCAT_DELIMITER)
    #print(concatenated)

    # serial
    #for inputFile in inputFiles:
    #    translateMsgFile(inputFile, *fnargs) 
    #
    # multiprocess
    #with multiprocessing.Pool(processes=numJobs) as pool:
    #    for inputFile in inputFiles:
    #        pool.apply_async(
    #            translateMsgFile, (inputFile, outputFileOrDir, model, temperature))
    #    pool.close()
    #    pool.join()
    #
    # threaded
    fnargs = (inputEncoding, outputFileOrDir, model, temperature)
    with concurrent.futures.ThreadPoolExecutor(max_workers=numJobs) as executor:
        futures = [
            executor.submit(translateAndVerifyMsgFile, inputFile, *fnargs)
            #executor.submit(translateMsgFile, inputFile, *fnargs)
            for inputFile in inputFiles]
        concurrent.futures.wait(futures)

              
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
