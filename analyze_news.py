import os
import json
from typing_extensions import TypedDict, Annotated, List
import operator
from datetime import datetime

from langchain.messages import AnyMessage, HumanMessage , AIMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel

# INIT LOGGING
console = Console()

# CONFIGURATION
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please set it before running.")

MODEL_NAME = "llama-3.3-70b-versatile"
EVALUATION_MODEL_NAME = "llama-3.1-8b-instant"
model = ChatGroq(model_name=MODEL_NAME, api_key=GROQ_API_KEY)
evaluation_model = ChatGroq(model_name=EVALUATION_MODEL_NAME, api_key=GROQ_API_KEY)
MAX_RETRIES = 3  # Maximum retry attempts per article
MIN_ARTICLE_LENGTH = 50  # Minimum characters for valid article
QUALITY_THRESHOLD = 0.8  # Minimum quality score to avoid optimization
ARTICLES_FILE = "articles.txt"
OUTPUT_FILE = "results.json"


# OUTPUT SCHEMAS
class Sentiment(BaseModel):
    overall: str = Field(..., description="إيجابي، سلبي، أو محايد")
    confidence: float = Field(..., ge=0.0, le=1.0)

class NewsCategory(BaseModel):
    category: str = Field(..., description="الفئة الرئيسية")
    subcategory: str = Field(..., description="الفئة الفرعية")

class ArticleAnalysis(BaseModel):
    SUMMARY: str
    PEOPLE: List[str]
    COUNTRIES: List[str]
    ORGANIZATIONS: List[str]
    LOCATIONS: List[str]
    SENTIMENT: Sentiment
    KEY_POINTS: List[str]
    NEWS_CATEGORY: NewsCategory
    ADDITIONAL_FOCUS: str = Field(..., description="نص يصف أي تركيز إضافي (يجب أن يكون نص واحد وليس قائمة)")

class QualityEvaluation(BaseModel):
    """Schema for evaluating analysis quality"""
    overall_quality: float = Field(..., ge=0.0, le=1.0, description="درجة الجودة الإجمالية")
    summary_accuracy: float = Field(..., ge=0.0, le=1.0, description="دقة الملخص")
    entity_completeness: float = Field(..., ge=0.0, le=1.0, description="اكتمال استخراج الكيانات")
    sentiment_appropriateness: float = Field(..., ge=0.0, le=1.0, description="مناسبة تحليل المشاعر")
    needs_improvement: bool = Field(..., description="هل يحتاج لتحسين؟")
    improvement_suggestions: str = Field(..., description="اقتراحات التحسين بالعربية")
    
    
# STATE MANAGEMENT
class AgentState(TypedDict):
    """State tracked through the LangGraph workflow"""
    current_article_index: int
    messages: Annotated[list[AnyMessage], operator.add]
    validated_data: dict
    quality_score: float
    quality_feedback: str
    needs_optimization: bool
    attempt_count: int
    final_result: dict
    all_results: List[dict]
    stats: dict


# UTILITY FUNCTIONS
def read_articles(filepath: str) -> List[str]:
    """Read articles from file, filtering out short snippets"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        articles = [
            art.strip() 
            for art in content.split('---') 
            if art.strip() and len(art.strip()) > MIN_ARTICLE_LENGTH
        ]
        
        if not articles:
            articles = [
                art.strip() 
                for art in content.split('\n\n\n') 
                if art.strip() and len(art.strip()) > MIN_ARTICLE_LENGTH
            ]
    
        return articles
    
    except FileNotFoundError:
        return []


# NODES
def analyzer(state: AgentState) -> dict:
    """Uses the LLM to process the current article and return the analysis."""
    current_article_index = state.get("current_article_index", 0)
    articles = read_articles(ARTICLES_FILE)
    current_article = articles[current_article_index]
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    console.print(f"[bold cyan] ANALYZER NODE - Processing Article #{current_article_index + 1}[/bold cyan]")
    
    system_prompt =  f"""أنت محلل أخبار خبير. قم بتحليل المقال الإخباري التالي وأعد النتائج بصيغة JSON فقط (بدون أي نص إضافي).
    يجب أن يحتوي الـ JSON على المفاتيح التالية بالإنجليزية، لكن القيم يجب أن تكون بالعربية:

    - SUMMARY: ملخص موجز للمقال.
    - PEOPLE: قائمة بالأشخاص المذكورين في المقال.
    - COUNTRIES: قائمة بالدول المذكورة في المقال.
    - ORGANIZATIONS: قائمة بالمنظمات المذكورة في المقال.
    - LOCATIONS: قائمة بالمواقع الجغرافية المذكورة في المقال.
    - SENTIMENT: كائن يحتوي على:
        • overall: إما "إيجابي"، "سلبي"، أو "محايد".
        • confidence: رقم بين 0 و1 يمثل ثقة التحليل.
    - KEY_POINTS: قائمة بأهم النقاط في المقال.
    - NEWS_CATEGORY: كائن يحتوي على:
        • category: الفئة الرئيسية للأخبار (مثل "سياسة"، "اقتصاد"، "رياضة").
        • subcategory: الفئة الفرعية (مثل "انتخابات"، "سوق الأسهم"، "كرة قدم").
    - ADDITIONAL_FOCUS: نص واحد (string) يصف أي تركيز إضافي - مهم جداً: يجب أن يكون نص واحد وليس قائمة.
    
    المقال:
    {current_article}

    الوقت الحالي:
    {current_time}

    تذكر: أرجع JSON فقط، بدون أي كلام قبله أو بعده."""
    
    try:
        result = model.with_structured_output(ArticleAnalysis).invoke(system_prompt)
        return {
            "messages": [HumanMessage(content=str(result))],
            "validated_data": result.model_dump()
        }
    except Exception as e:
        console.print(f"[red] Error in analyzer: {str(e)}[/red]")
        console.print(f"[yellow]Attempt {state.get('attempt_count', 0) + 1} failed[/yellow]")
        raise


def evaluator(state: AgentState) -> dict:
    """Evaluates the quality of the analysis"""
    current_article_index = state.get("current_article_index", 0)
    current_article = read_articles(ARTICLES_FILE)[current_article_index]
    
    console.print(f"[bold magenta] EVALUATOR NODE - Assessing Quality For Article #{current_article_index + 1} [/bold magenta]")

    last_message = state["messages"][-1] if state["messages"] else None
    analysis_text = last_message.content if last_message else ""
    
    evaluation_prompt = f"""أنت مُقيّم خبير لجودة تحليل الأخبار. قم بتقييم التحليل التالي للمقال الإخباري.

    المقال الأصلي:
    {current_article}

    التحليل المُقدم:
    {analysis_text}

    قم بتقييم التحليل بناءً على:
    1. دقة الملخص (هل يغطي النقاط الرئيسية؟)
    2. اكتمال استخراج الكيانات (الأشخاص، الدول، المنظمات، المواقع)
    3. مناسبة تحليل المشاعر
    4. ملاءمة التصنيف

    أعطِ:
    - overall_quality: درجة الجودة الإجمالية (0-1)
    - summary_accuracy: دقة الملخص (0-1)
    - entity_completeness: اكتمال الكيانات (0-1)
    - sentiment_appropriateness: مناسبة المشاعر (0-1)
    - needs_improvement: true فقط إذا كان overall_quality أقل من 0.75 أو هناك أخطاء كبيرة
    - improvement_suggestions: اقتراحات محددة للتحسين بالعربية (أو "لا توجد" إذا كان التحليل جيد)

    مهم: كن واقعياً - إذا كان overall_quality أعلى من 0.80، ضع needs_improvement = false

    أرجع JSON فقط."""
    
    try:
        # Add retry logic for intermittent API errors
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                result = evaluation_model.with_structured_output(QualityEvaluation).invoke(evaluation_prompt)
                console.print(f"[cyan] Quality Score: [bold]{result.overall_quality:.2f}[/bold][/cyan]")
                if result.improvement_suggestions:
                    console.print(f"   • Suggestions: [dim]{result.improvement_suggestions}[/dim]")
                
                return {
                    "messages": [HumanMessage(content=f"Quality Score: {result.overall_quality}")],
                    "quality_score": result.overall_quality,
                    "quality_feedback": result.improvement_suggestions,
                    "needs_optimization": result.needs_improvement
                }
            except Exception as retry_error:
                last_error = retry_error
                if attempt < max_retries - 1:
                    console.print(f"[yellow] Retry {attempt + 1}/{max_retries} due to API error...[/yellow]")
                    import time
                    time.sleep(0.5)
                else:
                    raise last_error
    except Exception as e:
        console.print(f"[red] Error in evaluator: {str(e)}[/red]")
        raise


def optimizer(state: AgentState) -> dict:
    """Optimizes the analysis based on quality feedback"""
    current_article_index = state.get("current_article_index", 0)
    current_article = read_articles(ARTICLES_FILE)[current_article_index]
    feedback = state.get("quality_feedback", "")
    attempt = state.get("attempt_count", 0)
    
    console.print(f"[bold yellow] OPTIMIZER NODE - Improving Analysis (Attempt {attempt + 1}) For Article #{current_article_index + 1}[/bold yellow]")
    
    optimization_prompt = f"""أنت محلل أخبار خبير. تم تقييم تحليلك السابق ووُجدت بعض النقاط التي تحتاج تحسين.

    المقال:
    {current_article}

    ملاحظات التحسين:
    {feedback}

    قم بإعادة التحليل مع الأخذ في الاعتبار هذه الملاحظات. أرجع JSON كامل بنفس الصيغة السابقة.
    
    مهم جداً: 
    - ADDITIONAL_FOCUS يجب أن يكون نص واحد (string) وليس قائمة (array)
    - مثال صحيح: "التركيز على الجوانب الإنسانية والطوارئ"
    - مثال خاطئ: ["الطوارئ", "الإطفاء"]

    تذكر: أرجع JSON فقط، بدون أي كلام قبله أو بعده."""
        
    try:
        result = model.with_structured_output(ArticleAnalysis).invoke(optimization_prompt)
        new_attempt = state.get("attempt_count", 0) + 1
        return {
            "messages": [HumanMessage(content=f"Optimized: {str(result)}")],
            "attempt_count": new_attempt,
            "validated_data": result.model_dump()
        }
    except Exception as e:
        console.print(f"[red] Error in optimizer: {str(e)}[/red]")
        raise


def should_optimize(state: AgentState) -> str:
    """Decides whether to optimize or finalize"""
    needs_opt = state.get("needs_optimization", False)
    attempt_count = state.get("attempt_count", 0)
    if needs_opt and attempt_count < MAX_RETRIES:
        console.print(f"[yellow]→ Routing to OPTIMIZER for improvement[/yellow]\n")
        return "optimize"
    
    return "finalize"


def finalize(state: AgentState) -> dict:
    """Finalize the analysis"""
    quality_score = state.get("quality_score", 0.0)
    attempt_count = state.get("attempt_count", 0)
    validated_data = state.get("validated_data", {})
    
    return {
        "messages": [HumanMessage(content="Analysis completed!")],
        "final_result": validated_data,
        "quality_score": quality_score,
        "attempt_count": attempt_count
    }


def reporter(state: AgentState) -> dict:
    """Reporter Node - Generates final statistics and saves results"""
    all_results = state.get("all_results", [])
    stats = state.get("stats", {})
    
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        console.print(f"[red] Error saving results: {str(e)}[/red]")
        return {"messages": [HumanMessage(content="Report failed!")]}
    
    console.print(f"\n{'='*70}")
    console.print("[bold cyan] FINAL ANALYSIS REPORT[/bold cyan]")
    console.print(f"{'='*70}\n")
    
    table = Table(show_header=True, header_style="bold magenta", border_style="cyan")
    table.add_column("Metric", style="cyan", width=40)
    table.add_column("Value", style="green", width=25)
    
    total = stats.get("total_articles", 0)
    successful = stats.get("successful", 0)
    failed = stats.get("failed", 0)
    total_opts = stats.get("total_optimizations", 0)
    quality_scores = stats.get("quality_scores", [])
    sentiment_scores = stats.get("sentiment_scores", [])
    processing_times = stats.get("processing_times", [])
    total_time = stats.get("total_time", 0.0)
    
    table.add_row("Total Articles", str(total))
    table.add_row(
        "Successfully Parsed", 
        f"{successful} ({successful/total*100:.1f}%)" if total > 0 else "0 (0.0%)"
    )
    table.add_row(
        "Failed / Invalid", 
        f"{failed} ({failed/total*100:.1f}%)" if total > 0 else "0 (0.0%)"
    )
    table.add_row("Total Optimizations", str(total_opts))
    
    if quality_scores:
        avg_quality = sum(quality_scores) / len(quality_scores)
        table.add_row("Average Quality Score", f"{avg_quality:.3f}")
    
    if sentiment_scores:
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        table.add_row("Average Sentiment Confidence", f"{avg_sentiment:.3f}")
    
    if total_opts > 0 and successful > 0:
        avg_opt = total_opts / successful
        table.add_row("Avg Optimizations/Article", f"{avg_opt:.2f}")
    
    # Add timing metrics
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        min_time = min(processing_times)
        max_time = max(processing_times)
        table.add_row("Total Processing Time", f"{total_time:.2f}s")
        table.add_row("Average Time/Article", f"{avg_time:.2f}s")
        table.add_row("Slowest Article", f"{max_time:.2f}s")
        table.add_row("Fastest Article", f"{min_time:.2f}s")
    
    console.print(table)
    
    # Success rate interpretation
    success_rate = (successful / total * 100) if total > 0 else 0
    
    console.print(f"\n{'='*70}")
    if success_rate == 100:
        console.print("[bold green] All articles processed successfully[/bold green]")
    elif success_rate >= 80:
        console.print("[bold yellow] Most articles processed successfully.[/bold yellow]")
    else:
        console.print("[bold red] Multiple failures occurred.[/bold red]")
    
    console.print(f"{'='*70}\n")
    console.print(f"[cyan] Results saved to: {OUTPUT_FILE}[/cyan]")
    console.print("[dim]Check the JSON file for detailed analysis of each article.[/dim]\n")
    
    return {
        "messages": [AIMessage(content="Report generated!")]
    }


# BUILD WORKFLOW
workflow = StateGraph(AgentState)
workflow.add_node("analyzer", analyzer)
workflow.add_node("evaluator", evaluator)
workflow.add_node("optimizer", optimizer)
workflow.add_node("finalize", finalize)

# DEFINE WORKFLOW EDGES
workflow.set_entry_point("analyzer")
workflow.add_edge("analyzer", "evaluator")
workflow.add_conditional_edges(
    "evaluator",
    should_optimize,
    {
        "optimize": "optimizer",
        "finalize": "finalize"
    }
)
workflow.add_edge("optimizer", "evaluator") 
agent = workflow.compile()

# MAIN EXECUTION
def analyze_all_articles():
    """Analyze all articles using the workflow"""
    
    articles = read_articles(ARTICLES_FILE)
    console.print(f"\n[green]✓[/green] Loaded {len(articles)} articles from {ARTICLES_FILE}\n")
    
    if not articles:
        console.print("[red] No articles found![/red]")
        return
    
    stats = {
        "total_articles": len(articles),
        "successful": 0,
        "failed": 0,
        "total_optimizations": 0,
        "quality_scores": [],
        "sentiment_scores": [],
        "processing_times": [],  
        "total_time": 0.0  
    }
    
    all_results = []
    start_time = datetime.now()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Analyzing articles...", total=len(articles))
        
        for idx in range(len(articles)):
            console.print(f"\n[bold]{'='*70}[/bold]")
            console.print(f"[bold cyan]Processing Article #{idx + 1}/{len(articles)}:[/bold cyan]")
            console.print(f"[bold]{'='*70}[/bold]")
            
            article_start_time = datetime.now()
            
            try:
                result = agent.invoke({
                    "messages": [],
                    "current_article_index": idx,
                    "attempt_count": 0
                })
                
                article_end_time = datetime.now()
                article_processing_time = (article_end_time - article_start_time).total_seconds()
                stats["processing_times"].append(article_processing_time)
                
                final_data = result.get("final_result")
                if not final_data:
                    final_data = result.get("validated_data")
                
                quality_score = result.get("quality_score", 0.0)
                optimization_cycles = result.get("attempt_count", 0)
                
                if final_data and isinstance(final_data, dict) and final_data.get("SUMMARY"):
                    result_with_metadata = {
                        "article_number": idx + 1,
                        "quality_score": quality_score,
                        "optimization_cycles": optimization_cycles,
                        "processing_time_seconds": round(article_processing_time, 2),
                        "SUMMARY": final_data.get("SUMMARY", ""),
                        "PEOPLE": final_data.get("PEOPLE", []),
                        "COUNTRIES": final_data.get("COUNTRIES", []),
                        "ORGANIZATIONS": final_data.get("ORGANIZATIONS", []),
                        "LOCATIONS": final_data.get("LOCATIONS", []),
                        "SENTIMENT": final_data.get("SENTIMENT", {"overall": "", "confidence": 0.0}),
                        "KEY_POINTS": final_data.get("KEY_POINTS", []),
                        "NEWS_CATEGORY": final_data.get("NEWS_CATEGORY", {"category": "", "subcategory": ""}),
                        "ADDITIONAL_FOCUS": final_data.get("ADDITIONAL_FOCUS", "")
                    }
                    
                    all_results.append(result_with_metadata)
                    stats["successful"] += 1
                    stats["quality_scores"].append(quality_score)
                    stats["total_optimizations"] += optimization_cycles
                    
                    if "SENTIMENT" in final_data:
                        stats["sentiment_scores"].append(final_data["SENTIMENT"]["confidence"])
                    
                    console.print(f"\n[green] Article #{idx + 1} completed successfully in {article_processing_time:.2f}s![/green]")
                else:
                    stats["failed"] += 1
                    console.print(f"\n[red] Article #{idx + 1} failed - no data returned[/red]")
                    
            except Exception as e:
                stats["failed"] += 1
                console.print(f"\n[red] Article #{idx + 1} failed with error: {str(e)}[/red]")
            
            progress.advance(task)
    
    end_time = datetime.now()
    stats["total_time"] = (end_time - start_time).total_seconds()
    
    reporter({
        "all_results": all_results,
        "stats": stats
    })


if __name__ == "__main__":
    analyze_all_articles()