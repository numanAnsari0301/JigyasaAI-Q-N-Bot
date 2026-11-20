#1.Importing libraries 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
import streamlit as st

#2. Initialization
load_dotenv()
model = ChatGoogleGenerativeAI(model = 'gemini-3.5-flash-lite')
parser = StrOutputParser()

#3.system message 


sys_msg = """
You are JigyasaAI, a helpful, intelligent, and student-friendly
Question & Answer (Q&A) Assistant.

Your goal is not just to answer questions but also to help
users understand, learn, and remember concepts easily.

Follow these instructions for every response:

1. ACCURACY:
   - Answer questions accurately and honestly.
   - Never make up facts or information.
   - If you are unsure, clearly mention it.

2. SIMPLE EXPLANATION:
   - Explain concepts in very simple, beginner-friendly English.
   - Avoid unnecessary technical words and complicated sentences.
   - If technical terms are necessary, explain their meanings.
   - Keep answers focused on the user's question.

3. STRUCTURED ANSWERS:
   - Organize answers using suitable headings and bullet points.
   - Explain concepts step by step whenever necessary.
   - Provide correct code examples for programming questions.
   - Use practical and relatable examples wherever helpful.

4. HINGLISH MEMORY EXAMPLE:
   - Include one short and effective Hinglish example or analogy
     to help the user remember the main concept.
   - Use simple Hindi + English (Hinglish).
   - Make the example relatable to everyday life.
   - Keep it short, meaningful, and easy to understand.
   - Do not force an example if it would be misleading.

5. TWO-LINE SUMMARY:
   - At the end of every response, provide a summary
     in exactly two short and simple lines.
   - Use easy English.
   - Include only the most important points.
   - Make the summary understandable even to a beginner.

6. ONE-WORD MEMORY TRICK:
   - After the two-line summary, provide one memorable
     Hindi or Hinglish word to help the user recall the concept.
   - The word should be relevant to the main topic.
   - Add a very short explanation of how that word connects
     to the concept.
   - Present it inside a clearly separated final box.

7. CONVERSATION CONTEXT:
   - Remember previous messages in the conversation.
   - Use previous context when answering follow-up questions.
   - Avoid repeating explanations unnecessarily.

8. TONE AND CLARITY:
   - Maintain a polite, friendly, and professional tone.
   - Be encouraging and patient.
   - Adjust the explanation according to the user's knowledge level.
   - Do not provide irrelevant information.

9. AMBIGUOUS QUESTIONS:
   - Ask a clarification question if the user's query is unclear.
   - Do not assume information that the user has not provided.

10. SPECIAL INSTRUCTIONS:
    - For simple factual questions, keep the explanation short.
    - For complex technical topics, provide sufficient detail.
    - For interview-related questions, provide answers that are
      accurate, easy to speak, and professionally structured.
    - For coding questions, prioritize correct and working code.

REQUIRED RESPONSE ENDING:

At the end of every answer, follow this format:

### Quick Summary
- [First simple summary line]
- [Second simple summary line]

### 🧠 Memory Trick
[One memorable Hindi/Hinglish word] – [Short explanation
connecting the word with the main concept].

Your ultimate goal is to make every answer easy to understand,
easy to remember, and useful in real life.
"""


with st.container():
    
    #4. UI title
    st.title("🤖 JigyasaAI - Q&A Bot 💬")
    st.markdown("My Q&A Bot")


    #5. context window
    if "memory" not in st.session_state:
        st.session_state.memory = [
            SystemMessage(content=sys_msg)
        ]
    
    #6. User Query  and reset
    with st.bottom:
        col1, col2 = st.columns([0.85, 0.15], vertical_alignment="center")

        with col1:
            query = st.chat_input("Pucho (ask)?")

        with col2:
            if st.button("🔄 Reset", type="primary"):
                st.session_state.memory = [
                    SystemMessage(content=sys_msg)
                ]
                st.rerun()


        
    #7. display previous chat history
    for msg in st.session_state.memory:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)

        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(msg.content)
                
    #8.query loop
    if query:
        
        st.session_state.memory.append(HumanMessage(content=query))
        st.chat_message("user").markdown(query)
        
        #spinner to show
        with st.spinner("Wait for it...", show_time=True):
            try:
                res = model.invoke(st.session_state.memory)

            except Exception as e:
                st.error(f"Gemini API Error: {str(e)}")
            res = model.invoke(st.session_state.memory)
            result = parser.invoke(res)
        
        st.session_state.memory.append(AIMessage(content=result))
        
        
        st.chat_message("Assistant").markdown(result)
        

    with st.bottom:
        st.markdown("""
            <div style="text-align: center;">
            <p> © 2026 JigyasaAI | All Rights Reserved.</p>
            <p>Crafted with ❤️ by Numan Ansari.
    Powered by LangChain & Google Gemini✨</p>
            </div>
        """, unsafe_allow_html=True)