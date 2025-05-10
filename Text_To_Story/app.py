from flask import Flask, render_template, request
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")

@app.route("/", methods=["GET", "POST"])
def index():
    story = ""
    title = ""
    word_count = 0
    prompt = ""

    if request.method == "POST":
        prompt = request.form["prompt"]
        story = generate_story(prompt)
        title = generate_title(prompt)
        word_count = len(story.split())

    return render_template("index.html", story=story, title=title, word_count=word_count, prompt=prompt)

def generate_story(prompt):
    story_prompt = f"Write a complete, short and imaginative story (120-180 words) based on the idea: '{prompt}'. Do not explain or describe writing. Just tell the story:\n\nStory:"
    story = generator(
        story_prompt,
        max_length=250,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.9,
        truncation=True,
        pad_token_id=50256
    )
    return story[0]['generated_text'].replace(story_prompt, "").strip()



def generate_title(prompt):
    try:
        title_prompt = f"{prompt.strip()}\nTitle:"
        title_output = generator(title_prompt, max_length=50, num_return_sequences=1)
        
        print("Generated Output:", title_output)
        
        generated_text = title_output[0]['generated_text']
        
        # Get text after "Title:"
        if "Title:" in generated_text:
            title = generated_text.split("Title:")[1].strip().split("\n")[0]
        else:
            title = generated_text.strip().split("\n")[0]
        
        return title if title else "Hloooooo"
    
    except Exception as e:
        print(f"Error generating title: {e}")
        return "Hlo"





if __name__ == "__main__":
    app.run(debug=True)
