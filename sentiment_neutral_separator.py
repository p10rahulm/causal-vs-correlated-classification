from utilities import get_claude_response


def prompt_builder(sentence):
    user_prompt = f"""
        You are given the following movie review :\n\n 
        {sentence} 

        Out goal is to separate this into two sets, 
        "sentiment_phrases": Those words in the movie review that are caused by the sentiment.
        "neutral_phrases": Those words in the movie review that are not caused by the sentiment.

        For example: 
        Sentence: "The movie was great. The main actors were Tom Cruise and Nicole Kidman. I wish to see it again."
        "sentiment_phrases": [ "The movie was great.", "I wish to see it again."]
        "neutral_phrases": ["The main actors were Tom Cruise and Nicole Kidman."]


        The template for the output should appear like this:

        {{
            "sentiment_phrases": [String, String, String],
            "neutral_phrases": [String, String]
        }}

        Task: Separate The sentence given: {sentence} into sentiment and neutral phrases in the given json format.

        IMPORTANT: Your response must be ONLY valid JSON that exactly matches the structure of the Template JSON. Do not include any explanatory text before or after the JSON. Begin your response with the opening curly brace '{{' and end with the closing curly brace '}}'.
        """
    return user_prompt


if __name__ == "__main__":
    sentence = """
    One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. They are right, as this is exactly what happened with me.<br /><br />The first thing that struck me about Oz was its brutality and unflinching scenes of violence, which set in right from the word GO. Trust me, this is not a show for the faint hearted or timid. This show pulls no punches with regards to drugs, sex or violence. Its is hardcore, in the classic use of the word.<br /><br />It is called OZ as that is the nickname given to the Oswald Maximum Security State Penitentary. It focuses mainly on Emerald City, an experimental section of the prison where all the cells have glass fronts and face inwards, so privacy is not high on the agenda. Em City is home to many..Aryans, Muslims, gangstas, Latinos, Christians, Italians, Irish and more....so scuffles, death stares, dodgy dealings and shady agreements are never far away.<br /><br />I would say the main appeal of the show is due to the fact that it goes where other shows wouldn't dare. Forget pretty pictures painted for mainstream audiences, forget charm, forget romance...OZ doesn't mess around. The first episode I ever saw struck me as so nasty it was surreal, I couldn't say I was ready for it, but as I watched more, I developed a taste for Oz, and got accustomed to the high levels of graphic violence. Not just violence, but injustice (crooked guards who'll be sold out for a nickel, inmates who'll kill on order and get away with it, well mannered, middle class inmates being turned into prison bitches due to their lack of street skills or prison experience) Watching Oz, you may become comfortable with what is uncomfortable viewing....thats if you can get in touch with your darker side.
    """
    user_prompt = prompt_builder(sentence)
    response = get_claude_response(user_prompt)
    if response:
        print(json.dumps(response, indent=2))
    else:
        print("Failed to get a valid response from the API.")
