# Class Exercise: Building a RAG System with Mistral

## Step 1: Create an Agentic/Assistant System Prompt

Choose a specific business context and create a system prompt that gives Mistral a professional role. This system prompt will define how the AI behaves and what expertise it has.

**Instructions:**

- Pick a realistic business or organization
- Choose a specific role/expertise for the AI (marketing expert, technical consultant, etc.)
- Create a system prompt that defines the AI's personality and knowledge area
- This will be used throughout the assignment for generating content

```python
# TODO: Choose your business and role
# Examples:
# - "TechStart Solutions - AI Consulting Firm" with role "AI Solutions Consultant"
# - "Green Energy Corp - Solar Installation Company" with role "Solar Energy Expert"
# - "HealthTech Plus - Medical Software Company" with role "Healthcare IT Specialist"


# Begin writing Python codes here
```

---

## Step 2: Generate Business Database Content

Use Mistral to create a comprehensive Q&A database for your chosen business. You'll prompt Mistral to generate realistic question-answer pairs that customers might ask about your services, pricing, processes, and expertise.

**Instructions:**

- Use your system prompt from Step 1 to give Mistral the business context
- Create a prompt asking Mistral to generate 10-15 Q&A pairs for your business
- Ask for questions covering different topics: services, pricing, processes, technical details, contact info
- Format should be clear (Q: question, A: answer)
- Parse the generated text into a usable list of dictionaries

```python
# TODO: Generate Q&A database using Mistral
# You need to:
# 1. Set up the Mistral model (use the pipeline approach from the original notebook)
# 2. Create a function to get clean responses from Mistral
# 3. Write a prompt asking Mistral to generate business Q&A pairs
# 4. Parse the generated text into a list of dictionaries with 'question' and 'answer' keys
# 5. Display your generated Q&A pairs clearly


# Begin writing Python codes here
```

---

## Step 3: Implement FAISS Vector Database

Convert your Q&A database into embeddings (numerical vectors) and store them in a FAISS index for fast similarity search. This allows users to ask questions and quickly find the most relevant information from your knowledge base.

**Instructions:**

- Install and import sentence-transformers for creating embeddings
- Convert all your questions into numerical vectors using an embedding model
- Create a FAISS index to store these vectors for fast similarity search
- Implement a search function that can find similar questions based on user input
- Test your search functionality with a sample query

```python
# TODO: Implement FAISS Vector Database
# You need to:
# 1. Install sentence-transformers: !pip install sentence-transformers faiss-cpu
# 2. Import SentenceTransformer and faiss
# 3. Load an embedding model (e.g., 'distilbert-base-uncased-distilled-squad')
# 4. Extract questions and answers from your Q&A database
# 5. Convert questions to embeddings using the model
# 6. Create a FAISS index and add the embeddings
# 7. Create a search function that takes a user question and returns similar Q&A pairs
# 8. Test the search function with a sample query

# Begin writing Python codes here
```

---

## Step 4: Create Test Questions

Generate two types of questions to test your RAG system: questions that CAN be answered from your database (answerable) and questions that CANNOT be answered (unanswerable). This tests how well your system knows its limitations.

**Instructions:**

- Use Mistral to generate 5 questions that your business CAN answer (about your services, pricing, processes, etc.)
- Use Mistral to generate 5 questions that your business CANNOT answer (competitor info, unrelated topics, personal details, etc.)
- Extract the questions from the generated text into clean lists
- These will test whether your RAG system correctly identifies when it can and cannot provide good answers

```python
# TODO: Create Test Questions
# You need to:
# 1. Generate ANSWERABLE questions using Mistral (questions your business can answer)
# 2. Generate UNANSWERABLE questions using Mistral (questions outside your expertise)
# 3. Parse both sets of questions into clean lists
# 4. Display both types of questions clearly
# 5. Make sure you have at least 5 questions of each type

# Begin writing Python codes here
```

---

## Step 5: Implement and Test Questions

Run both types of questions through your RAG system and analyze how well it distinguishes between questions it can answer well versus questions it cannot answer reliably.

**Instructions:**

- Test your answerable questions - they should get high similarity scores with your database
- Test your unanswerable questions - they should get low similarity scores
- Set a similarity threshold to determine "can answer" vs "cannot answer"
- Analyze the performance: did answerable questions score high? Did unanswerable questions score low?
- Calculate accuracy rates for both question types

```python
# TODO: Test Your RAG System
# You need to:
# 1. Create a testing function that searches your database for each question
# 2. Set a similarity threshold (e.g., 0.7) to determine good vs poor matches
# 3. Test all answerable questions and count how many are correctly identified as answerable
# 4. Test all unanswerable questions and count how many are correctly identified as unanswerable
# 5. Calculate and display performance statistics
# 6. Show examples of good and poor matches
```

---

## Step 6: Model Experimentation and Ranking

Test multiple Q&A models from Hugging Face and rank them based on performance, speed, and confidence scores.

**Instructions:**

- Test the 4 required models plus 2 additional models of your choice
- Evaluate each model on speed, confidence scores, and answer quality
- Rank models from best to worst with clear explanations
- Identify which models provide good confidence scores while maintaining reasonable output
- Compare performance across different question types

```python
# TODO: Test and Rank QA Models
# Required models to test:
# - "consciousAI/question-answering-generative-t5-v1-base-s-q-c"
# - "deepset/roberta-base-squad2"
# - "google-bert/bert-large-cased-whole-word-masking-finetuned-squad"
# - "gasolsun/DynamicRAG-8B"
# Plus 2 additional QA models of your choice
#
# You need to:
# 1. Set up QA pipelines for each model
# 2. Test them with your questions and retrieved contexts
# 3. Measure response time and confidence scores
# 4. Rank models based on composite performance
# 5. Identify models with good confidence handling
# 6. Explain why each model ranked where it did
```

### Model Ranking Explanation

```
Write your explanation here:


```

---

## Learning Objectives

By completing this exercise, you will:

1. **Understand Prompt Engineering**: Learn how to create effective system prompts for business-specific AI assistants
2. **Implement RAG Systems**: Build a complete RAG system with vector database and similarity search
3. **Evaluate AI Models**: Compare and rank multiple QA models based on performance metrics
4. **Understand System Limitations**: Test how well your system knows what it can and cannot answer

---

**IST402 - AI Agents & RAG Systems**

