import json
import numpy as np
import random
import os

def primary_emotions_concept_dataset(data_dir, user_tag='', assistant_tag='', seed=0):
    random.seed(0)

    template_str = '{user_tag} Consider the {emotion} of the following scenario:\nScenario: {scenario}\nAnswer: {assistant_tag} '
    emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
    raw_data = {}
    for emotion in emotions:
        with open(os.path.join(data_dir, f'{emotion}.json')) as file:
            # raw_data[emotion] = json.load(file)
            raw_data[emotion] = list(set(json.load(file)))[:200]

    formatted_data = {}
    for emotion in emotions:
        c_e, o_e = raw_data[emotion], np.concatenate([v for k,v in raw_data.items() if k != emotion])
        random.shuffle(o_e)

        data = [[c,o] for c,o in zip(c_e, o_e)]
        train_labels = []
        for d in data:
            true_s = d[0]
            random.shuffle(d)
            train_labels.append([s == true_s for s in d])
        
        data = np.concatenate(data).tolist()
        data_ = np.concatenate([[c,o] for c,o in zip(c_e, o_e)]).tolist()
        
        emotion_test_data = [template_str.format(emotion=emotion, scenario=d, user_tag=user_tag, assistant_tag=assistant_tag) for d in data_]
        emotion_train_data = [template_str.format(emotion=emotion, scenario=d, user_tag=user_tag, assistant_tag=assistant_tag) for d in data]

        formatted_data[emotion] = {
            'train': {'data': emotion_train_data, 'labels': train_labels},
            'test': {'data': emotion_test_data, 'labels': [[1,0]* len(emotion_test_data)]}
        }
    return formatted_data

def primary_emotions_function_dataset(data_dir, user_tag='', assistant_tag='', seed=0):
    random.seed(0)

    train_template_str = '{user_tag} Act as if you are extremely {emo}. {assistant_tag} {scenario}' 
    emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
    with open(os.path.join(data_dir, "all_truncated_outputs.json"), 'r') as file:
        all_truncated_outputs = json.load(file)
    
    emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
    emotions_adj = [
        ("joyful", "happy", "cheerful"), 
        ("sad", "depressed", "miserable"),
        ("angry", "furious", "irritated"),
        ("fearful", "scared", "frightened"),
        ("disgusted", "sicken", "revolted"), 
        ("surprised", "shocked", "astonished")
    ]
    emotions_adj_ant = [
        ("dejected", "unhappy", "dispirited"), 
        ("cheerful", "optimistic", "happy"),
        ("pleased", "calm", "peaceful"),
        ("fearless", "bold", "unafraid"),
        ("approved", "delighted", "satisfied"), 
        ("unimpressed", "indifferent", "bored")
    ]

    formatted_data = {}
    for emotion, emotion_adj, emotion_adj_ant in zip(emotions, emotions_adj, emotions_adj_ant):
        emotion_train_data_tmp = [[
            train_template_str.format(emo=np.random.choice(emotion_adj), scenario=s, user_tag=user_tag, assistant_tag=assistant_tag), 
            train_template_str.format(emo=np.random.choice(emotion_adj_ant), scenario=s, user_tag=user_tag, assistant_tag=assistant_tag)
        ] for s in all_truncated_outputs]
        
        train_labels = []
        for d in emotion_train_data_tmp:
            true_s = d[0]
            random.shuffle(d)
            train_labels.append([s == true_s for s in d])

        emotion_train_data = np.concatenate(emotion_train_data_tmp).tolist()

        formatted_data[emotion] = {
            'train': {'data': emotion_train_data, 'labels': train_labels},
        }
    return formatted_data

def primary_persona_concept_dataset_test(data_dir, user_tag='', assistant_tag='', seed=0):
    random.seed(0)

#     template_str = '{user_tag} What is the personality of the author for following text:\nText: {scenario} \
# {assistant_tag} The personality of the author is'
#     template_str = '{user_tag} Consider 2 personality traits: A and B. A and B are codenames for 2 opposing personality traits. The authors will either be A or B. \
# Based on this, what is the personality of the author for following text: {scenario}.\n\
# After answer A or B with the tag <answer>.{assistant_tag} The personality of the author is <answer>'
#     template_str = '{user_tag} Consider the personality trait:{personality}/{personality_anti}. What is the personality of the author for following text:\nText: {scenario}.\n\
# Answer {personality} or {personality_anti}.{assistant_tag} The personality of the author is <answer>'
    template_str = '{user_tag} Consider the personality trait:{personality}/{personality_anti}. What is the personality of the author for following text:\nText: {scenario}.\n\
{assistant_tag} The personality of the author is '
#     template_str = '{user_tag} The Myers-Briggs Type Indicator (MBTI) is a popular personality assessment tool that \
# categorizes individuals into 16 different personality types based on four key dichotomies. One of these dichotomies is Extraversion (E) vs. Introversion (I), which describes where people \
# primarily focus their attention and get their energy from.\n\nExtraversion (E):\nEnergy Source: Extraverts gain energy from external stimuli, such as social interactions and \
# engaging with the outside world.\n\nFocus: Their focus is outward, toward people, activities, and things. They often feel energized and motivated by being around others.\nCommunication: \
# They tend to be more talkative, expressive, and assertive. Extraverts often think out loud and enjoy engaging in conversations and activities with others.\nSocial Preference: Extraverts \
# typically enjoy being in groups, meeting new people, and participating in social activities. They might seek out social interactions to recharge.\n\nIntroversion (I):\nEnergy Source: \
# Introverts gain energy from internal stimuli, such as thoughts, reflections, and solitary activities.\nFocus: Their focus is inward, toward their inner thoughts and feelings. \
# They often feel energized and refreshed by spending time alone or with a small, close group of people.\nCommunication: Introverts tend to be more reserved, reflective, and deliberate \
# in their communication. They may prefer to think things through before speaking and often express themselves better in writing than in conversation.\nSocial Preference: Introverts \
# generally prefer more intimate settings and one-on-one interactions. Large groups or prolonged social activities may feel draining to them.\n\nConsider the MBTI personality trait:{personality}/{personality_anti}, \
# given by the description above, what is the personality of the author for following text:\nText: {scenario} {assistant_tag} The personality of the author is'
    personalities = ["introversion"]
    personalities_anti = ["extraversion"]
    raw_data = {}
    for personality, personality_anti in zip(personalities, personalities_anti):
        with open(os.path.join(data_dir, f'{personality}.json')) as file:
            # raw_data[emotion] = json.load(file)
            raw_data[personality] = list(set(json.load(file)))[:500]
        with open(os.path.join(data_dir, f'{personality_anti}.json')) as file:
            # raw_data[emotion] = json.load(file)
            raw_data[personality_anti] = list(set(json.load(file)))[:500]

    formatted_data = {}
    for personality, personality_anti in zip(personalities, personalities_anti):
        c_e, o_e = raw_data[personality], raw_data[personality_anti]
        random.shuffle(o_e)

        data = [[c,o] for c,o in zip(c_e, o_e)]
        train_labels = []
        for d in data:
            true_s = d[0]
            random.shuffle(d)
            train_labels.append([s == true_s for s in d])
        
        data = np.concatenate(data).tolist()
        data_ = np.concatenate([[c,o] for c,o in zip(c_e, o_e)]).tolist()
        
        emotion_test_data = [template_str.format(personality=personality, personality_anti=personality_anti, scenario=d, user_tag=user_tag, assistant_tag=assistant_tag) for d in data_]
        emotion_train_data = [template_str.format(personality=personality, personality_anti=personality_anti, scenario=d, user_tag=user_tag, assistant_tag=assistant_tag) for d in data]

        formatted_data[personality] = {
            'train': {'data': emotion_train_data, 'labels': train_labels},
            'test': {'data': emotion_test_data, 'labels': [[1,0]* len(emotion_test_data)]}
        }
    return formatted_data

def primary_persona_concept_dataset_test2(data_dir, user_tag='', assistant_tag='', seed=0):
    random.seed(0)

    template_str = '{user_tag} Consider the personality trait:{personality}/{personality_anti}. What is the personality of the author for following text:\nText: {scenario}.\n\
{assistant_tag} The personality of the author is '
#     template_str = '{user_tag} What is the personality of the author for following text:\nText: {scenario}.\n\
# {assistant_tag} The personality of the author is '
#     template_str = '{user_tag} Consider the personality trait:{personality}/{personality_anti}. What is the personality of the author for following text:\nText: {scenario}.\n\
# Answer {personality} or {personality_anti} with the tag <answer>.{assistant_tag} The personality of the author is ' 
#     template_str = '{user_tag} The Myers-Briggs Type Indicator (MBTI) is a popular personality assessment tool that \
# categorizes individuals into 16 different personality types based on four key dichotomies. One of these dichotomies is Extraversion (E) vs. Introversion (I), which describes where people \
# primarily focus their attention and get their energy from.\n\nExtraversion (E):\nEnergy Source: Extraverts gain energy from external stimuli, such as social interactions and \
# engaging with the outside world.\n\nFocus: Their focus is outward, toward people, activities, and things. They often feel energized and motivated by being around others.\nCommunication: \
# They tend to be more talkative, expressive, and assertive. Extraverts often think out loud and enjoy engaging in conversations and activities with others.\nSocial Preference: Extraverts \
# typically enjoy being in groups, meeting new people, and participating in social activities. They might seek out social interactions to recharge.\n\nIntroversion (I):\nEnergy Source: \
# Introverts gain energy from internal stimuli, such as thoughts, reflections, and solitary activities.\nFocus: Their focus is inward, toward their inner thoughts and feelings. \
# They often feel energized and refreshed by spending time alone or with a small, close group of people.\nCommunication: Introverts tend to be more reserved, reflective, and deliberate \
# in their communication. They may prefer to think things through before speaking and often express themselves better in writing than in conversation.\nSocial Preference: Introverts \
# generally prefer more intimate settings and one-on-one interactions. Large groups or prolonged social activities may feel draining to them.\n\nConsider the MBTI personality trait:{personality}/{personality_anti}, \
# given by the description above, what is the personality of the author for following text:\nText: {scenario} {assistant_tag} The personality of the author is '
    # personalities = ["introversion"]
    # personalities_anti = ["extraversion"]
    # p_fine = ["INFJ","INFP","INTJ","INTP","ISFJ","ISFP","ISTJ","ISTP"]
    # p_fine_anti = ["ENFJ","ENFP","ENTJ","ENTP","ESFJ","ESFP","ESTJ","ESTP"]
    personalities = ["judgement"]
    personalities_anti = ["perception"]
    p_fine = ["INTJ","INFJ","ENTJ","ENFJ","ISTJ","ISFJ","ESTJ","ESFJ"]
    p_fine_anti = ["INTP","INFP","ENTP","ENFP","ISTP","ISFP","ESTP","ESFP"]
    raw_data = {}
    for personality, personality_anti in zip(personalities, personalities_anti):
        raw_data[personality] = []
        raw_data[personality_anti] = []
        for p, p_anti in zip(p_fine,p_fine_anti):
            with open(os.path.join(data_dir, f'{p}.json')) as file1:
                # raw_data[emotion] = json.load(file)
                d1 =  list(set(json.load(file1)))
                raw_data[personality].extend(d1[:500])
            with open(os.path.join(data_dir, f'{p_anti}.json')) as file2:
                # raw_data[emotion] = json.load(file)
                d2 =  list(set(json.load(file2)))
                raw_data[personality_anti].extend(d2[:500])
            if len(d1) <500 or len(d2) <500:
                print("Error")
            # max=200
            # if len(d1) >= len(d2):
            #     raw_data[personality_anti].extend(d1[:len(d2)])
            #     raw_data[personality].extend(d2)
            # else:
            #     raw_data[personality_anti].extend(d1)
            #     raw_data[personality].extend(d2[:len(d1)])
    formatted_data = {}
    for personality, personality_anti in zip(personalities, personalities_anti):
        c_e, o_e = raw_data[personality], raw_data[personality_anti]
        # random.shuffle(o_e)

        data = [[c,o] for c,o in zip(c_e, o_e)]
        train_labels = []
        for d in data:
            true_s = d[0]
            random.shuffle(d)
            train_labels.append([s == true_s for s in d])
        
        data = np.concatenate(data).tolist()
        data_ = np.concatenate([[c,o] for c,o in zip(c_e, o_e)]).tolist()
        
        emotion_test_data = [template_str.format(personality=personality, personality_anti=personality_anti, scenario=d, user_tag=user_tag, assistant_tag=assistant_tag) for d in data_]
        emotion_train_data = [template_str.format(personality=personality, personality_anti=personality_anti, scenario=d, user_tag=user_tag, assistant_tag=assistant_tag) for d in data]

        formatted_data[personality] = {
            'train': {'data': emotion_train_data, 'labels': train_labels},
            'test': {'data': emotion_test_data, 'labels': [[1,0]* len(emotion_test_data)]}
        }
    return formatted_data