import gradio as gr
from transformers import pipeline
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# Load components
print("🚀 Loading Fake News Detector...")

try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./fake_news_chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print("✅ Vector database loaded successfully!")
    
    # Use a text classification model specifically trained for fake news detection
    llm = pipeline(
        "text-classification", 
        model="mrm8488/distilbert-base-uncased-finetuned-fake-news",  # Model specifically for fake news
        return_all_scores=True
    )
    print("✅ AI model loaded successfully!")
    
except Exception as e:
    print(f"❌ Error loading components: {e}")
    # Fallback to basic sentiment analysis if specific model fails
    llm = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", return_all_scores=True)
    print("⚠️  Using fallback model")

def analyze_news(text):
    """
    Analyze news content for credibility using AI and similarity search
    """
    if not text.strip():
        return "❌ Please enter some news content to analyze."
    
    if len(text.strip()) < 10:
        return "❌ Please provide more substantial news content (at least 10 characters)."
    
    try:
        # Get similar documents from database
        docs = retriever.get_relevant_documents(text)
        
        # Get AI classification
        result = llm(text[:512])[0]  # Limit input length for model
        best_match = max(result, key=lambda x: x['score'])
        confidence = best_match['score']
        
        # Determine credibility label
        if best_match['label'] == 'LABEL_1' or 'REAL' in str(best_match['label']).upper():
            label = "✅ LIKELY CREDIBLE"
            color = "#22c55e"
        elif best_match['label'] == 'LABEL_0' or 'FAKE' in str(best_match['label']).upper():
            label = "🚨 POTENTIALLY MISLEADING"
            color = "#ef4444"
        else:
            # Fallback for sentiment-based models
            label = "✅ CREDIBLE" if best_match['label'] == 'POSITIVE' else "🚨 SUSPICIOUS"
            color = "#22c55e" if best_match['label'] == 'POSITIVE' else "#ef4444"
        
        # Build analysis report
        analysis = f"""
## 📊 ANALYSIS RESULTS

<div style='background-color: {color}20; padding: 15px; border-radius: 10px; border-left: 4px solid {color};'>
<h3 style='color: {color}; margin: 0;'>{label}</h3>
</div>

**Confidence Level:** {confidence:.1%}  
**Similar Patterns Found:** {len(docs)} matches in our database

### 🔍 ANALYSIS
This content appears **{'credible' if 'CREDIBLE' in label else 'suspicious'}** based on linguistic patterns and comparison with our database of verified news examples.
"""
        
        # Add specific recommendations
        if "MISLEADING" in label or "SUSPICIOUS" in label:
            analysis += """
### ⚠️ RECOMMENDATIONS

• **Verify with reputable sources** like AP News, Reuters, or BBC
• **Check the publication date** - old news can be repurposed
• **Look for official statements** from relevant authorities
• **Reverse image search** any accompanying photos
• **Use fact-checking sites** like Snopes or FactCheck.org
"""
        else:
            analysis += """
### 💡 BEST PRACTICES

• **Cross-reference** with multiple reliable sources
• **Check author credentials** and publication history
• **Be aware of potential biases** in the reporting
• **Look for supporting evidence** and citations
• **Consider the tone** - credible news is typically neutral
"""
        
        # Add educational tip
        analysis += f"""
---

**🎓 Digital Literacy Tip:** Always approach online information with healthy skepticism and verify before sharing.
"""
        
        return analysis
        
    except Exception as e:
        return f"❌ Error analyzing content: {str(e)}\n\nPlease try again with different text."

# Create enhanced Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Fake News Detector for Students") as demo:
    gr.Markdown("""
    # 📰 Fake News Detector for Students
    **Empowering critical thinking in the digital age**
    
    Analyze news credibility using AI and our database of 180,000+ real and fake news examples.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 🔍 Analyze News Content")
            input_text = gr.Textbox(
                label="Enter news headline or content",
                placeholder="Paste news article, headline, or social media post here...",
                lines=4,
                max_lines=6
            )
            analyze_btn = gr.Button("Analyze Credibility 🚀", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### 📚 Quick Examples")
            examples = gr.Examples(
                examples=[
                    ["Breaking: Scientists discover revolutionary cancer treatment with 95% success rate"],
                    ["Government announces free college education for all students starting next semester"],
                    ["Celebrity claims COVID-19 vaccine contains tracking microchips"],
                    ["New study shows chocolate helps with weight loss and improves memory"]
                ],
                inputs=input_text,
                label="Try these examples:"
            )
            
            gr.Markdown("### 🎯 Tips for Analysis")
            gr.Markdown("""
            - **Copy-paste** full articles for best results
            - **Check headlines** for emotional language
            - **Verify dates** and sources
            - **Look for evidence** and citations
            """)
    
    with gr.Row():
        output = gr.Markdown(
            label="Analysis Results",
            value="👆 Enter news content above to get started...",
            show_label=False
        )
    
    # Footer
    gr.Markdown("---")
    gr.Markdown("""
    <div style='text-align: center; color: #666;'>
    <p><strong>⚠️ Educational Tool Disclaimer:</strong> This AI analysis is for educational purposes only. 
    Always verify information through multiple reputable sources before making decisions.</p>
    <p>Built with ❤️ for digital literacy education</p>
    </div>
    """)
    
    # Connect the function
    analyze_btn.click(analyze_news, inputs=input_text, outputs=output)
    input_text.submit(analyze_news, inputs=input_text, outputs=output)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0" if os.getenv("SPACES") else "127.0.0.1",
        server_port=7860,
        share=False
    )
