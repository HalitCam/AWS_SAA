**AWS Semantic Quiz Bot Link:** [**https://examchatbot.streamlit.app/**](https://examchatbot.streamlit.app/)

This Python application is an interactive "AWS Semantic Quiz Bot." The program parses pdf file(having 1300 questions) to automatically extract questions, answers, and explanations. The web interface allows user to start a quiz (by topic or random), provides instant feedback on their answers, shows the correct explanations, and displays a final score upon completion.

**What Happens After You Run It? (Step-by-Step Results)**

Here is what you will see in your browser tab when you run the application:

**a. Main Screen (Topic Selection)**

Once indexing is complete, the application's main screen will appear:

- **Title:** "AWS Semantic Quiz Bot 🧠☁️"
- **Info:** "X adet AWS sorusu indekslendi." (X AWS questions indexed.) - It will display the number of questions it successfully parsed from the PDF.
- **Text Box:** "Which topic do you want to be quizzed on?" (Leave blank for random)
- **Number Box:** "How many questions?" (Defaults to 3).
- **Button:** "Start Quiz"

**b. The Quiz Starts**

When you press the "Start Quiz" button, two scenarios can occur:

- **Scenario A (You entered a topic):** If you typed a topic like "S3 and storage" into the text box, the application will show a loading animation saying "Finding the 3 best questions about 'S3 and storage'...". It will then find the 3 questions in the database that are semantically **closest** to this topic and start the quiz.
- **Scenario B (You left the topic blank):** If you leave the box empty, it will say "Selecting 3 random questions..." and will select 3 **random** questions from the PDF.

**c. Question Answering Screen**

- A title like "Question 1 / 3" will appear on the screen.
- The text of the question will be displayed.
- Below it, the options for that question will be listed as radio buttons.
- You select an option and press the "Submit Answer" button.

**d. Answer Check**

When you press the button:

- If your answer is **correct:** It will display "Correct! 🎉".
- If your answer is **incorrect:** It will display "Incorrect. The correct answer was: \[Correct answer text\]".
- In either case, if the question has an explanation, this explanation will be shown in a blue info box as "Explanation: ...".
- The "Next Question" button will appear to move to the next question.

This loop continues until you have completed the number of questions you selected (e.g., 3).

**e. Results Screen**

When you have answered all the questions:

- Celebratory balloons will float across the screen.
- The title "Quiz Finished!" will appear.
- You will see a result text, such as "You answered 2 out of 3 questions correctly."
- The "Start Over" button will appear to begin again. If you press this, everything resets, and you return to the Main Screen (Topic Selection).

**Note:**

You may sometimes encounter unexpected formatting errors. For example, the explanation for a question might appear after the text of the final answer choice. This is a system-generated error. If you encounter any issues, please take a screenshot and send it. And regardless, please continue with your exam.
