const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType,
  PageBreak, LevelFormat
} = require('docx');
const fs = require('fs');

// ── Helpers ───────────────────────────────────────────────────────────────────
const BLUE       = "1F4E79";
const LIGHT_BLUE = "2E75B6";
const TEAL       = "006064";
const CONTENT_W  = 9360;

const h1 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_1,
  children: [new TextRun({ text, bold: true, size: 36, font: "Arial", color: BLUE })],
  spacing: { before: 480, after: 240 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: LIGHT_BLUE, space: 4 } },
});

const h2 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2,
  children: [new TextRun({ text, bold: true, size: 28, font: "Arial", color: LIGHT_BLUE })],
  spacing: { before: 320, after: 160 },
});

const h3 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_3,
  children: [new TextRun({ text, bold: true, size: 24, font: "Arial", color: TEAL })],
  spacing: { before: 200, after: 100 },
});

const p = (text) => new Paragraph({
  children: [new TextRun({ text, font: "Arial", size: 22 })],
  spacing: { after: 160 },
  alignment: AlignmentType.JUSTIFIED,
});

const pb = (normal, bold) => new Paragraph({
  children: [
    new TextRun({ text: bold, font: "Arial", size: 22, bold: true }),
    new TextRun({ text: " " + normal, font: "Arial", size: 22 }),
  ],
  spacing: { after: 140 },
});

const note = (text) => new Paragraph({
  children: [new TextRun({ text: "💡 " + text, font: "Arial", size: 20, italics: true, color: "555555" })],
  spacing: { after: 140 },
  indent: { left: 360 },
});

const warn = (text) => new Paragraph({
  children: [new TextRun({ text: "⚠️ " + text, font: "Arial", size: 20, bold: true, color: "C00000" })],
  spacing: { after: 140 },
  indent: { left: 360 },
});

const code = (text) => new Paragraph({
  children: [new TextRun({ text, font: "Courier New", size: 18, color: "1F4E79" })],
  spacing: { after: 60 },
  indent: { left: 720 },
  shading: { fill: "EEF4FF", type: ShadingType.CLEAR },
});

const bl = (text) => new Paragraph({
  numbering: { reference: "bullets", level: 0 },
  children: [new TextRun({ text, font: "Arial", size: 22 })],
  spacing: { after: 100 },
});

const blb = (bold, normal) => new Paragraph({
  numbering: { reference: "bullets", level: 0 },
  children: [
    new TextRun({ text: bold, font: "Arial", size: 22, bold: true }),
    new TextRun({ text: " — " + normal, font: "Arial", size: 22 }),
  ],
  spacing: { after: 100 },
});

const pageBreak = () => new Paragraph({ children: [new PageBreak()] });

const spacer = () => new Paragraph({ children: [new TextRun("")], spacing: { after: 120 } });

// Table builder
const mkTable = (headers, rows) => {
  const colW = Math.floor(CONTENT_W / headers.length);
  const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
  const borders = { top: border, bottom: border, left: border, right: border };

  const mkCell = (text, isHeader, colIdx) => new TableCell({
    borders,
    width: { size: colW, type: WidthType.DXA },
    shading: {
      fill: isHeader ? "1F4E79" : (colIdx % 2 === 0 ? "F2F8FF" : "FFFFFF"),
      type: ShadingType.CLEAR
    },
    margins: { top: 80, bottom: 80, left: 160, right: 160 },
    children: [new Paragraph({
      children: [new TextRun({
        text: String(text || ""),
        font: "Arial", size: 20,
        bold: isHeader,
        color: isHeader ? "FFFFFF" : "000000",
      })],
    })],
  });

  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: headers.map(() => colW),
    rows: [
      new TableRow({ children: headers.map((h, i) => mkCell(h, true, i)) }),
      ...rows.map(row => new TableRow({ children: row.map((c, i) => mkCell(c, false, i)) })),
    ],
  });
};

// ════════════════════════════════════════════════════════════════════════════
// TITLE PAGE
// ════════════════════════════════════════════════════════════════════════════

const titlePage = [
  new Paragraph({ spacing: { before: 1200 } }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "LOAN COLLECTION ANALYTICS", bold: true, size: 56, font: "Arial", color: "1F4E79" })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 160 },
    children: [new TextRun({ text: "AI-Powered Prompt-to-Visualization System", size: 32, font: "Arial", color: "2E75B6" })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 480 },
    border: { top: { style: BorderStyle.SINGLE, size: 4, color: "2E75B6" }, bottom: { style: BorderStyle.SINGLE, size: 4, color: "2E75B6" } },
    children: [new TextRun({ text: "Complete Learning & Reference Guide — 9 Documents", bold: true, size: 26, font: "Arial", color: "333333" })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 480 },
    children: [new TextRun({ text: "Everything you need to explain, defend, and demo this project", size: 24, font: "Arial", color: "555555", italics: true })],
  }),
  pageBreak(),
];

// ════════════════════════════════════════════════════════════════════════════
// DOC 1 — WHAT IS THIS PROJECT AND WHY DOES IT EXIST
// ════════════════════════════════════════════════════════════════════════════

const doc1 = [
  h1("Document 1 — What Is This Project and Why Does It Exist"),
  p("Before we talk about any technology, we need to understand the business problem this project solves. Every technical decision — every library, every model, every file — exists to solve this specific problem. If you understand the problem deeply, every technical choice will make sense."),

  h2("1.1 The Business Problem"),
  p("A loan collection company manages thousands of loan accounts every month. At the end of each month, they get a massive dataset — in our case, 89,255 rows — with information about every loan. Each row tells you things like: did the customer pay their EMI? Did they bounce? How many days overdue are they? Which field executive visited them?"),
  p("The collections manager needs to answer questions like:"),
  bl("Which branches have the highest bounce rate this month?"),
  bl("Which team leaders are getting the best visit coverage?"),
  bl("How many loans moved from the 30-59 DPD bucket into NPA?"),
  bl("Which portfolio is performing worst?"),
  spacer(),
  p("Before this system existed, answering these questions required:"),
  bl("Opening Power BI and manually configuring filters"),
  bl("Knowing which columns to use (technical knowledge required)"),
  bl("Choosing the right chart type yourself"),
  bl("Writing DAX queries or SQL manually"),
  bl("Waiting for an analyst to prepare the report"),
  spacer(),
  p("The collections manager is a business person, not a data analyst. They know what they want to see but not how to get it. This creates a bottleneck — every data question requires a technical person in the loop."),

  h2("1.2 The Solution This System Provides"),
  p("This system removes that bottleneck completely. The collections manager types a plain English question — exactly as they would say it to a colleague — and the system automatically:"),
  bl("Understands what metric they want to see"),
  bl("Asks 2-3 follow-up questions to narrow down the filters"),
  bl("Writes the database query automatically"),
  bl("Runs the query against the real database"),
  bl("Picks the most appropriate chart type"),
  bl("Generates 3-5 business insights from the result"),
  spacer(),
  p("The user never writes a single line of SQL. They never configure a filter. They never choose a chart type. The AI handles all of that."),

  h2("1.3 Why This Is Hard to Build"),
  p("This sounds simple but it is genuinely difficult for several reasons:"),
  pb("Language is ambiguous.", "When someone says 'show me bad loans', do they mean NPA accounts? Loans in 30-59 DPD? Write-offs? The system must figure out the correct interpretation."),
  pb("Column names are not user language.", "The database column is called op_bucket. The user says 'risk category'. The system must bridge this gap without the user knowing technical terms."),
  pb("SQL must be exactly right.", "A small error — wrong column name, missing quote, wrong aggregation — and the query fails. There is no room for approximation."),
  pb("Charts must match the data.", "A line chart for 134 branches is unreadable. A heatmap for time series data is wrong. The system must choose based on what the data actually looks like, not just what the question says."),
  pb("Everything runs locally.", "No cloud APIs, no sending company data to external servers. The AI models must run on the company's own hardware."),

  h2("1.4 What Makes This Project Different from Just Using ChatGPT"),
  p("This is a question you will definitely be asked. Here is the precise answer:"),
  mkTable(
    ["Aspect", "ChatGPT / Cloud AI", "This System"],
    [
      ["Data privacy", "Your data goes to OpenAI servers", "All data stays on company hardware"],
      ["Database access", "Cannot connect to your private DB", "Direct connection to PostgreSQL"],
      ["Domain knowledge", "Knows nothing about your columns", "Knows all 43 columns, business logic, bucket rules"],
      ["SQL accuracy", "Guesses column names", "Uses exact snake_case column names from schema"],
      ["Chart selection", "Generic suggestions", "Rule-based logic from actual DataFrame analysis"],
      ["Cost per query", "API cost per token", "Zero — runs on local GPU/CPU"],
    ]
  ),

  pageBreak(),
];

// ════════════════════════════════════════════════════════════════════════════
// DOC 2 — THE TECHNOLOGY STACK — WHAT EACH TOOL IS AND WHY WE CHOSE IT
// ════════════════════════════════════════════════════════════════════════════

const doc2 = [
  h1("Document 2 — The Technology Stack: What Each Tool Is and Why We Chose It"),
  p("Every technology in this project was chosen deliberately over a specific alternative. This document explains what each tool is from first principles, then explains why we chose it over the alternatives we considered."),

  h2("2.1 Streamlit — The User Interface"),
  h3("What it is"),
  p("Streamlit is a Python library that turns Python scripts into web applications automatically. You write normal Python code — define variables, call functions, create charts — and Streamlit renders it as a webpage that users can interact with. You do not write HTML, CSS, or JavaScript."),
  h3("Why we chose it over alternatives"),
  mkTable(
    ["Alternative", "Why Rejected"],
    [
      ["Flask/Django", "Requires writing HTML templates, JavaScript, CSS — 10x more code for same result"],
      ["Dash (Plotly)", "More complex setup, steeper learning curve, overkill for this use case"],
      ["Jupyter Notebook", "Not a real web app — cannot be shared as a URL, not production-ready"],
      ["React + FastAPI", "Professional but requires two codebases, JavaScript knowledge, much longer to build"],
    ]
  ),
  h3("Key concept: How Streamlit works"),
  p("Streamlit reruns the entire Python script from top to bottom on every user interaction. Every button click, every dropdown selection — the whole script runs again. This is why we needed the two-phase pipeline design. The pipeline state must survive these reruns, which is why we store everything in st.session_state."),
  warn("This rerun behavior is the #1 source of bugs in Streamlit apps. The reset bug we fixed was caused by not understanding this."),

  h2("2.2 Ollama — Running AI Models Locally"),
  h3("What it is"),
  p("Ollama is software that downloads AI language models and serves them through a local API. Think of it as the bridge between the Python code and the AI model files. Without Ollama, you would need to load model weights manually into memory, handle GPU allocation, and build an inference server yourself — weeks of work."),
  p("Ollama exposes a simple API at http://localhost:11434. Our Python code sends a POST request with a prompt and gets back the model's response. It handles all the complexity of GPU memory management, model loading, and token generation internally."),
  h3("Why not use cloud APIs?"),
  bl("Security: Company loan data cannot leave the network. Every query contains customer information."),
  bl("Cost: At scale, GPT-4 API costs can be significant. Local models have zero per-query cost."),
  bl("Control: We can use any model without depending on a third-party service being available."),
  h3("How we use Ollama in the code"),
  code("POST http://localhost:11434/api/chat"),
  code('{ "model": "qwen2.5-coder:14b", "messages": [...], "stream": false }'),
  p("The ollama_client.py file handles all communication with Ollama, including retry logic when the model times out."),

  h2("2.3 Qwen2.5-Coder 14B — The Primary AI Model"),
  h3("What it is"),
  p("Qwen2.5-Coder is an AI language model made by Alibaba Cloud, specialized for code generation tasks. The '14B' means it has 14 billion parameters — parameters are the numerical weights that define how the model thinks. More parameters generally means more capable reasoning, at the cost of more memory and slower responses."),
  p("This model is used for the three most complex tasks: understanding the user's question (intent analysis), generating clarifying questions, and writing the PostgreSQL SQL query."),
  h3("Why Qwen Coder over other models"),
  mkTable(
    ["Model", "Parameters", "Decision"],
    [
      ["Qwen2.5-Coder 14B", "14B", "CHOSEN — best balance of code quality and hardware fit"],
      ["Qwen2.5-Coder 7B", "7B", "Tried — JSON quality dropped significantly on complex domain logic"],
      ["Qwen2.5-Coder 32B", "32B", "Rejected — requires 24GB VRAM, not available on demo hardware"],
      ["Llama 3.3 70B", "70B", "Rejected — too large, extremely slow on CPU"],
      ["DeepSeek-Coder-V2", "16B", "Good at SQL but weaker domain reasoning for loan-specific metrics"],
    ]
  ),
  h3("Why a Coder model specifically?"),
  p("We need the model to generate structured outputs — JSON and SQL — not creative writing. Coder-specialized models are fine-tuned on code datasets, which makes them better at producing syntactically correct structured output consistently. A base language model (even a large one) tends to add explanations, use wrong syntax, or wrap output in markdown."),

  h2("2.4 Qwen2.5 7B — The Secondary AI Model"),
  h3("What it is"),
  p("The base Qwen 7B model (not the Coder variant) handles lighter tasks: deciding which chart type to use and generating business insights from the data. These tasks require common sense reasoning about data presentation, not code generation expertise."),
  h3("Why a smaller model for these tasks?"),
  p("Chart type decision and insight generation are simpler classification/writing tasks. Using the 14B Coder model for these would take 5-10 minutes per call. The 7B model handles them in 30-60 seconds. We save time without losing quality for these specific tasks."),

  h2("2.5 PostgreSQL in Docker — The Database"),
  h3("What PostgreSQL is"),
  p("PostgreSQL (often called 'Postgres') is a relational database — it stores data in tables with rows and columns, just like Excel but designed for millions of rows and multiple simultaneous users. It uses SQL (Structured Query Language) to query data."),
  p("Our loan collection data lives in a table called 'loan_dashboard' inside PostgreSQL, with 43 columns and 89,255 rows per monthly snapshot."),
  h3("What Docker is"),
  p("Docker is a tool that packages software into 'containers' — self-contained units that include everything the software needs to run. Instead of installing PostgreSQL manually on the computer (which involves configuration, path setup, service registration), we run one Docker command and the database is ready."),
  code("docker run -d --name loan-db -e POSTGRES_PASSWORD=yourpassword -p 5432:5432 postgres"),
  p("This command downloads PostgreSQL, creates the database server, and starts it on port 5432. If the computer restarts, the container can be restarted with one command. If something breaks, delete the container and recreate it in seconds."),
  h3("Why Docker over installing PostgreSQL directly?"),
  bl("Portability: The same Docker command works on Windows, Mac, and Linux"),
  bl("Isolation: PostgreSQL runs in its own environment, cannot conflict with other software"),
  bl("Easy cleanup: Delete the container and all traces of the database are gone"),
  bl("Demo-friendly: Can be set up and torn down quickly for demonstrations"),

  h2("2.6 ChromaDB — The Knowledge Store"),
  h3("What it is"),
  p("ChromaDB is a vector database. Before explaining what that means, we need to understand the problem it solves."),
  p("We have 5 JSON files containing knowledge about our data: column definitions, business logic, bucket rules, SQL patterns, and chart selection rules. Combined, these files contain about 15,000 tokens of text — too large to send to the AI model with every query."),
  p("ChromaDB stores these as 'vectors' (mathematical representations of meaning) and retrieves only the most relevant chunks for each specific question. When the user asks about bounce rate, ChromaDB finds and returns only the bounce-related sections — about 1,400 tokens instead of 15,000."),
  h3("What a vector embedding is"),
  p("An embedding is a list of numbers that represents the meaning of a piece of text. Similar meanings produce similar number patterns. 'bounce rate' and 'payment failure' produce similar embeddings because they mean related things. 'loan age' and 'MOB bucket' produce similar embeddings. ChromaDB uses these similarity patterns to find relevant content."),
  h3("What all-MiniLM-L6-v2 is"),
  p("This is the embedding model — it converts text into vectors. It is a small model (80MB) that runs locally, produces 384-dimensional vectors, and is fast enough to embed queries in real-time. We chose it because it runs without any internet connection and produces good retrieval quality for this domain."),

  h2("2.7 SQLAlchemy — The Database Connection Layer"),
  h3("What it is"),
  p("SQLAlchemy is a Python library that manages database connections. Instead of opening a new connection to PostgreSQL for every query (which takes 50-200ms each time), SQLAlchemy maintains a 'connection pool' — a set of pre-opened connections that queries can reuse immediately."),
  p("We configured a pool of 3 persistent connections with 2 overflow connections. This means up to 5 queries can run simultaneously without waiting for connection setup."),
  h3("Why SQLAlchemy over psycopg2 directly?"),
  p("psycopg2 is the low-level PostgreSQL driver — it handles the actual network protocol. SQLAlchemy wraps psycopg2 and adds connection pooling, which is critical for performance. Without pooling, each of our 5 pipeline steps would spend 100ms just opening a connection before running the query."),

  h2("2.8 Plotly — The Charting Library"),
  h3("What it is"),
  p("Plotly is a Python charting library that creates interactive charts. 'Interactive' means users can hover over data points to see exact values, zoom in, pan, and download the chart as an image — all without any additional code from us."),
  h3("Why Plotly over Matplotlib/Seaborn"),
  mkTable(
    ["Feature", "Matplotlib/Seaborn", "Plotly"],
    [
      ["Interactivity", "Static image — no hover, no zoom", "Full hover, zoom, pan, download"],
      ["Streamlit integration", "st.pyplot() — static", "st.plotly_chart() — interactive"],
      ["File size", "PNG image ~50-200KB", "JSON/HTML ~50KB, renders in browser"],
      ["Mobile", "Fixed size image", "Responsive, resizes to screen"],
      ["Export", "Save PNG manually", "Built-in PNG/HTML export button"],
    ]
  ),
  note("We kept Matplotlib/Seaborn in requirements.txt for the PDF export feature, which requires static images."),

  pageBreak(),
];

// ════════════════════════════════════════════════════════════════════════════
// DOC 3 — THE PIPELINE — HOW A QUESTION BECOMES A CHART
// ════════════════════════════════════════════════════════════════════════════

const doc3 = [
  h1("Document 3 — The Pipeline: How a Question Becomes a Chart"),
  p("This is the heart of the system. Understanding this pipeline means understanding exactly what happens at every step between the user typing a question and seeing a chart. Every file, every function, every model call fits into this sequence."),

  h2("3.1 Overview of the Two-Phase Architecture"),
  p("The pipeline is split into two phases. This is not a design preference — it is a technical requirement forced by how Streamlit works."),
  p("Streamlit reruns the entire Python script on every button click. If we had a single pipeline function that showed filter widgets mid-execution and then waited for the user to click 'Generate Dashboard', the second button click would restart the script from scratch, wiping out all the work done in Phase 1."),
  p("The solution: Phase 1 ends by saving all its results to session state and calling st.rerun(). Phase 2 reads those saved results and continues. The session state acts as the memory that survives across reruns."),

  h2("3.2 Phase 1 — Understanding the Question"),
  h3("Step 1: Intent Analysis (models/analyzer.py)"),
  p("The user's plain English question goes to Qwen Coder 14B with a carefully engineered system prompt. The system prompt contains:"),
  bl("The role of the model (loan collection analyst)"),
  bl("All 14 available metric keys with exact names the model must use"),
  bl("All 43 column names in snake_case format"),
  bl("Business rules (bounce = Tech + Non Tech only, NEVER PAID)"),
  bl("ChromaDB context chunks most relevant to this question (~1,400 tokens)"),
  spacer(),
  p("The model returns a JSON object with these fields:"),
  code('{ "metric": "Bounce Rate", "metric_key": "Bounce_Percent",'),
  code('  "columns_needed": ["bounce_status", "cust_id", "branch"],'),
  code('  "group_by": "branch",'),
  code('  "aggregation": "DISTINCT_COUNT",'),
  code('  "slicer_candidates": ["Region", "Branch", "Next Date"],'),
  code('  "confidence": "High" }'),
  spacer(),
  p("After parsing, the group_by field is normalised through a FIELD_TO_SNAKE dictionary. If Qwen returns 'Branch' (display name), it becomes 'branch' (DB column name). This normalisation happens even if the model ignores the instructions."),
  h3("Step 2: Clarifying Questions (models/clarifier.py)"),
  p("Using the IntentResult from Step 1, Qwen Coder 14B generates 2-4 targeted questions. These are not fixed questions shown for every query — they are dynamically generated based on which slicers are relevant to this specific metric."),
  p("For a bounce rate question, the model generates: Which region? Which portfolio? Which month?"),
  p("For a coverage question, it generates: Which team leader? Which branch? Which month?"),
  p("The questions are rendered as Streamlit widgets (selectbox or multiselect). Answers are saved to session state as pending_slicers = {'region': 'Pune', 'portfolio_new': 'All', ...}."),

  h2("3.3 Phase 2 — Getting the Data and Showing It"),
  h3("Step 3: SQL Generation (models/sql_generator.py)"),
  p("This is the most critical step — where errors are most costly. The SQL generator takes the IntentResult and the user's slicer answers and produces a PostgreSQL SELECT query."),
  p("The slicer answers go through the FIELD_TO_DB translation map first. Display names like 'Op bucket' become 'op_bucket', 'Cust ID' becomes 'cust_id', 'Allocation 1' becomes 'allocation_1'. This is what was causing the column-not-found errors."),
  p("The system prompt tells Qwen to use only snake_case column names. The WHERE clause is built from translated slicer values. The GROUP BY uses the normalised group_by from the IntentResult."),
  p("After Qwen returns SQL, it is validated: must start with SELECT or WITH, must not contain INSERT/UPDATE/DELETE/DROP. If validation fails, the rule-based fallback generates correct SQL for all 8 metric types without calling Qwen again."),
  h3("Step 4: Database Fetch (database/client.py)"),
  p("The SQL runs against PostgreSQL through the SQLAlchemy connection pool. Results come back as a pandas DataFrame. The _clean_columns() method in client.py then renames all snake_case columns back to display names for the UI:"),
  code("op_bucket → Op bucket"),
  code("cust_id → Cust ID"),
  code("bounce_status → Bounce status"),
  p("This bidirectional translation — display names to snake_case for SQL generation, snake_case to display names for chart rendering — is the key architectural pattern that makes the whole system work."),
  h3("Step 5: Chart Decision (models/chart_decider.py)"),
  p("The chart decider uses a two-layer approach. The first layer is pure rules — no AI involved:"),
  bl("If metric_key is in the METRIC_CHART_RULES dictionary, use that chart type directly"),
  bl("If the DataFrame has 2 categorical columns + 1 numeric: heatmap"),
  bl("If 1 categorical column with many unique values: horizontal bar"),
  bl("If 2 numeric columns: scatter (only if both are truly independent)"),
  bl("If 1 numeric column only: KDE distribution"),
  spacer(),
  p("Qwen 7B then validates this decision. If Qwen's suggestion violates a hard rule (scatter for categorical data, line for 134 branches), the rule-based decision is kept. This is why we call it 'rule-first, model-validates' — not 'model decides'."),
  h3("Step 6: Chart Rendering (charts/renderer.py + chart files)"),
  p("The renderer receives a ChartConfig dictionary and the DataFrame. It routes to the correct chart module based on chart_type. Each chart module (bar.py, heatmap.py, etc.) creates a Plotly figure which Streamlit renders with st.plotly_chart()."),
  h3("Step 7: Insights (models/insight_generator.py)"),
  p("Qwen 7B receives a text summary of the data — top 5 rows, bottom 3 rows, min/max/avg for each numeric column. It generates 3-5 business observations in plain English. The rule-based fallback uses pandas operations to always generate something meaningful even if the model fails."),

  h2("3.4 The State Machine"),
  p("The pipeline phase is stored as a string in session state:"),
  mkTable(
    ["Phase Value", "Meaning", "What Happens"],
    [
      ["None", "Home screen", "Show question input and examples"],
      ["'clarifying'", "Phase 1 complete", "Show filter widgets + Generate button"],
      ["'done'", "Phase 2 complete", "Show dashboard: KPIs + slicers + chart + insights"],
    ]
  ),

  pageBreak(),
];

// ════════════════════════════════════════════════════════════════════════════
// DOC 4 — THE DATA — WHAT EVERY COLUMN MEANS
// ════════════════════════════════════════════════════════════════════════════

const doc4 = [
  h1("Document 4 — The Data: What Every Column Means"),
  p("This document explains the loan collection dataset in detail. Understanding the data is essential — the AI models only know this because we told them through the context JSON files. If someone asks you what 'Op bucket' means or why we use DISTINCTCOUNT for customer metrics, this document has the answer."),

  h2("4.1 What This Dataset Is"),
  p("The loan_dashboard table contains one row per loan application per monthly snapshot. Each month, the collection team loads 89,255 rows covering all active loans. The data tracks:"),
  bl("Which risk category (bucket) the loan is in at the start and end of the month"),
  bl("Whether the EMI payment bounced"),
  bl("Whether the field executive visited the customer"),
  bl("Financial details: outstanding principal, total overdue"),
  bl("Team hierarchy: which FE, TL, SH is managing this loan"),

  h2("4.2 The Critical Granularity Rule"),
  p("This is the most important technical concept in the data:"),
  warn("One customer can have MULTIPLE loan applications. Always use COUNT(DISTINCT cust_id) for customer-level metrics, not COUNT(loanappno)."),
  p("Example: A customer has 2 home loans. If both bounce, loanappno count = 2, but cust_id count = 1. Bounce rate should be reported at customer level (1 customer bounced), not loan level (2 loans bounced). The AI models are explicitly told this rule multiple times in every prompt."),

  h2("4.3 The Bucket System Explained"),
  p("The most important concept in loan collection is the bucket system. Buckets classify loans by how overdue they are:"),
  mkTable(
    ["Bucket Name", "DB Value", "What It Means", "Who Is In It"],
    [
      ["Current", "Current", "Fully healthy, paying on time", "dpd=0, bounce_status='PAID', no EMI increase"],
      ["Risk X", "Risk X", "DPD is zero but showing warning signs", "dpd=0 but bounce was Tech/Non Tech OR EMI increased"],
      ["1-29 DPD", "1-29 DPD", "1 to 29 days overdue", "dpd between 1 and 29"],
      ["30-59 DPD", "30-59 DPD", "30 to 59 days overdue", "dpd between 30 and 59"],
      ["60-89 DPD", "60-89 DPD", "60 to 89 days overdue — pre-NPA", "dpd between 60 and 89, risk_npa=0"],
      ["NPA", "NPA", "Non-Performing Asset — more than 90 days overdue", "dpd > 90 OR (60-89 DPD with risk_npa=1)"],
      ["Write-off", "Write-off", "Removed from books, still collecting", "Cases written off from company books"],
    ]
  ),

  h2("4.4 Bucket Movement — cust_wise_status"),
  p("One of the most powerful metrics is how a customer's bucket changed between the start and end of the month. This is stored in cust_wise_status, derived by comparing op_bucket (opening) and closing_bucket (closing):"),
  mkTable(
    ["Status Value", "Meaning", "Example Movement"],
    [
      ["Current", "Was Current and stayed Current", "Current → Current"],
      ["Norm", "IMPROVED — moved to a better bucket", "30-59 DPD → Risk X, or NPA → Current"],
      ["Flow", "WORSENED — moved to a worse bucket", "1-29 DPD → 30-59 DPD, or Risk X → 30-59 DPD"],
      ["Stab", "STABLE — stayed in same bucket", "30-59 DPD → 30-59 DPD"],
      ["Roll Back", "PARTIALLY IMPROVED — moved up one bucket", "30-59 DPD → 1-29 DPD"],
      ["Risk NPA", "NEAR NPA — likely to become NPA next month", "60-89 DPD → NPA or 30-59 DPD → 60-89 DPD"],
      ["NPA", "Still NPA", "NPA → NPA"],
      ["Write-off", "Still written off", "Write-off → Write-off"],
    ]
  ),
  p("Resolution % = percentage of customers with cust_wise_status = 'Norm'. A collections team's primary goal is to maximize this number."),

  h2("4.5 Bounce Status Explained"),
  p("When a customer's EMI payment fails, it is called a bounce. There are three possible values:"),
  bl("PAID — The NACH (automatic bank debit) succeeded. Customer paid. NOT a bounce."),
  bl("Tech — Technical bounce: payment failed due to a bank/system error, not customer's fault"),
  bl("Non Tech — Non-technical bounce: customer intentionally did not maintain sufficient balance"),
  warn("PAID is NEVER counted as a bounce. This is stated in every system prompt because models sometimes include PAID in bounce counts. bounce_status IN ('Tech', 'Non Tech') is the only correct filter."),

  h2("4.6 Coverage Metrics Explained"),
  p("Coverage tracks how well the field team is visiting customers. It has specific logic:"),
  bl("Allocated: An FE (field executive) has been assigned to this loan (allocation_1 is not 'NA')"),
  bl("Visited: The FE actually visited the customer during this month (visit_or_not = 'Visited')"),
  bl("Coverage % = Visited AND Allocated / All Allocated — denominator is ONLY allocated accounts"),
  bl("Intensity = Average visits per customer = SUM(visit_count) / COUNT(DISTINCT cust_id)"),
  note("Unallocated accounts are excluded from Coverage % — they are the Sales team's responsibility, not Collection."),

  h2("4.7 The Column Name Problem and Why It Exists"),
  p("The PostgreSQL database stores columns in snake_case: op_bucket, cust_id, bounce_status. This is standard database convention — lowercase, underscores, no spaces."),
  p("The business users and all our context JSON files use display names: Op bucket, Cust ID, Bounce status. These are more readable for humans."),
  p("The system must translate between these two worlds constantly:"),
  bl("SQL generation uses snake_case (what the database expects)"),
  bl("Chart rendering uses display names (what the column rename map produces)"),
  bl("Clarifying questions use display names (what users understand)"),
  bl("Slicer answers from users use display names (Region, Branch)"),
  bl("These are translated to snake_case before building WHERE clauses"),

  pageBreak(),
];

// ════════════════════════════════════════════════════════════════════════════
// DOC 5 — CONTEXT ENGINEERING — HOW WE TEACH THE AI MODELS
// ════════════════════════════════════════════════════════════════════════════

const doc5 = [
  h1("Document 5 — Context Engineering: How We Teach the AI Models"),
  p("Context engineering is the practice of carefully designing what information gets sent to AI models and how it is structured. This is arguably the most important technical skill in this project. A model with perfect prompts and limited parameters will outperform a larger model with poor prompts."),

  h2("5.1 What Context Engineering Is"),
  p("AI language models have no memory between conversations. Every time we call the model, it starts fresh. We must provide everything it needs to know — in the right format, in the right order, with the right emphasis — in a single prompt."),
  p("The challenge: we have ~15,000 tokens of domain knowledge (column definitions, business logic, SQL patterns). The AI model can process up to ~128,000 tokens, but sending all 15,000 tokens every time would be:"),
  bl("Slow — more tokens = longer processing time"),
  bl("Expensive — wasted compute on irrelevant information"),
  bl("Less accurate — models lose focus when given too much irrelevant context"),
  p("Solution: ChromaDB retrieves only the ~1,400 most relevant tokens for each specific question."),

  h2("5.2 The 5 Context JSON Files"),
  p("Our domain knowledge is structured across 5 JSON files, each serving a specific purpose:"),
  mkTable(
    ["File", "Contents", "Goes To"],
    [
      ["01_schema_context.json", "All 43 columns: type, description, valid values, SQL usage hints, consumer language", "Both models"],
      ["02_business_logic.json", "Bucket rules, movement logic, metric formulas, granularity rules, null handling", "Qwen Coder 14B only"],
      ["03_language_mapping.json", "Consumer phrases → DB columns: 'bad loans' → op_bucket, 'bounced' → bounce_status", "Qwen Coder 14B only"],
      ["04_visualization_context.json", "Chart selection rules, metric-to-chart mapping, color palette guide, layout rules", "Qwen 7B only"],
      ["05_sql_patterns.json", "Ready-made SQL templates for all 14 metrics — Qwen adapts these instead of writing from scratch", "Qwen Coder 14B only"],
    ]
  ),

  h2("5.3 How ChromaDB Retrieval Works"),
  p("When the user asks 'show me bounced loans by branch', here is what happens:"),
  bl("The question is converted to a 384-dimensional vector by all-MiniLM-L6-v2"),
  bl("ChromaDB compares this vector against all 153 stored document vectors using cosine similarity"),
  bl("The 6 most similar documents from schema_context are returned (bounce-related fields)"),
  bl("The 5 most similar from business_logic (bounce metric formulas and rules)"),
  bl("The 4 most similar from language_mapping (phrase-to-field mappings for 'bounce')"),
  bl("The 4 most similar from sql_patterns (bounce SQL templates)"),
  bl("Total: ~1,400 tokens of focused, relevant context"),
  spacer(),
  p("Without ChromaDB, we would send all 15,000 tokens. The model would spend processing capacity on irrelevant coverage metrics and NPA rules that have nothing to do with the bounce question."),

  h2("5.4 System Prompt Design Principles"),
  p("Every system prompt in this project follows the same design principles, learned through iteration:"),
  h3("Principle 1: State hard rules explicitly and repeatedly"),
  p("The 'PAID is not a bounce' rule appears in the analyzer prompt, the SQL generator prompt, and the fallback SQL templates. Once is not enough — models drift toward the most common association (PAID = paid = good customer = include in count) without repeated reinforcement."),
  h3("Principle 2: Give exact values, not descriptions"),
  p("Instead of 'bounce status can be paid or bounced', we write: bounce_status IN ('Tech', 'Non Tech'). Exact SQL syntax the model can copy directly is more reliable than natural language descriptions the model must translate."),
  h3("Principle 3: Temperature 0.1 for structured outputs"),
  p("Temperature controls how creative/random the model's responses are. At temperature 1.0, the model might write 'bounce_Status' instead of 'bounce_status' — close but wrong. At temperature 0.1, it consistently follows the exact format specified. All our model calls use temperature=0.1."),
  h3("Principle 4: Tell the model what NOT to do"),
  p("'Return ONLY the SQL — no explanation, no markdown, no comments' eliminates the most common failure mode: the model wrapping SQL in markdown code fences or adding English explanations before the code block."),
  h3("Principle 5: Rule-based fallback for every AI step"),
  p("No matter how good the prompt is, the model will sometimes fail. Every step has a rule-based fallback that runs without calling the model. The analyzer falls back to keyword matching. The SQL generator has templates for all 8 metric types. The insight generator uses pandas operations."),

  h2("5.5 The JSON Parsing Problem"),
  p("AI models are instructed to return JSON, but they sometimes return:"),
  bl("JSON wrapped in markdown: ```json { ... } ```"),
  bl("JSON with an explanation before it: 'Here is the analysis: { ... }'"),
  bl("Slightly malformed JSON: single quotes instead of double quotes"),
  p("The parse_json_safely() function in helpers.py handles these cases by stripping markdown fences, finding the first { or [ character, and passing only that substring to json.loads(). The extract_json_from_text() function uses regex to find any JSON block anywhere in the model's response as a last resort."),

  pageBreak(),
];

// ════════════════════════════════════════════════════════════════════════════
// DOC 6 — THE CODE ARCHITECTURE — WHY EACH FILE EXISTS
// ════════════════════════════════════════════════════════════════════════════

const doc6 = [
  h1("Document 6 — The Code Architecture: Why Each File Exists"),
  p("This document explains the reasoning behind the folder structure and why code is organized the way it is. Understanding architecture means understanding why we separated things instead of putting everything in one file."),

  h2("6.1 The Layered Architecture"),
  p("The codebase is organized in layers, where each layer has one responsibility and depends only on layers below it:"),
  mkTable(
    ["Layer", "Files", "Responsibility"],
    [
      ["UI Layer", "app.py, ui/", "Streamlit rendering, session management, pipeline orchestration"],
      ["Models Layer", "models/", "AI model calls, prompt engineering, response parsing"],
      ["Charts Layer", "charts/", "Plotly figure creation and routing"],
      ["Database Layer", "database/", "PostgreSQL connection, query execution, mock data"],
      ["Core Layer", "core/", "ChromaDB, filter management, session state"],
      ["Utils Layer", "utils/", "Logging, benchmarking, shared helpers — no business logic"],
      ["Config Layer", "config.py, .env", "All constants and credentials — imported everywhere else"],
    ]
  ),
  p("The key rule: code in a higher layer can import from lower layers, but NEVER the reverse. chart files never import from models. models never import from ui. This prevents circular dependencies and makes each layer independently testable."),

  h2("6.2 Why utils/ Has No Business Logic"),
  p("logger.py, benchmark.py, and helpers.py contain zero domain knowledge. They know nothing about loans, bounces, or buckets. This is intentional — utilities should be reusable in any project."),
  p("If we put loan-specific code in helpers.py, we could not reuse helpers.py in a different project without carrying loan business logic with it. Separation keeps utility code clean."),

  h2("6.3 Why Each Model File Is Separate"),
  p("analyzer.py, clarifier.py, sql_generator.py, chart_decider.py, and insight_generator.py each handle exactly one pipeline step. This separation has concrete benefits:"),
  bl("Independent testing: You can test SQL generation without running intent analysis"),
  bl("Independent replacement: Swap the SQL generator for a different approach without touching the clarifier"),
  bl("Independent prompts: Each step has its own system prompt optimized for its specific task"),
  bl("Clear debugging: When something goes wrong, you know immediately which step failed"),
  p("The alternative — one giant 'pipeline.py' file — would make it impossible to know which part of the code is responsible for which failure."),

  h2("6.4 Why database/ Has Three Files"),
  h3("connection.py — The pool"),
  p("Only one thing: maintains the SQLAlchemy connection pool. No query logic, no business rules, just raw execute(sql) → DataFrame. This is tested by calling db.ping() and checking if SELECT 1 returns."),
  h3("client.py — The orchestrator"),
  p("Validates SQL, routes to real database or mock, cleans column names on the way out. Knows about the rename_map but not about the connection pool internals."),
  h3("mock.py — The simulator"),
  p("Generates realistic fake data without touching the database. Critical for demos and development. Seeded with numpy.random.seed(42) so the same question always returns the same mock data — essential for reproducible demos."),
  p("Why three files instead of one? Because you should be able to swap out any one piece. Replace mock.py with a different data generator. Replace connection.py with a cloud database connection. Neither change affects the other."),

  h2("6.5 Why Each Chart Type Has Its Own File"),
  p("bar.py, line.py, heatmap.py, kde.py, scatter.py, boxplot.py, treemap.py each contain one render() function. The renderer.py file just routes to the correct one."),
  p("The benefit: if the heatmap chart needs a fix, you open heatmap.py. You do not search through 800 lines of a combined charts file trying to find the heatmap section. Each file is 60-100 lines and completely readable in 5 minutes."),

  h2("6.6 The Singleton Pattern"),
  p("Almost every class in this project uses the singleton pattern: a module-level variable and a get_X() function that creates the object on first call and returns the same object on subsequent calls:"),
  code("_instance = None"),
  code("def get_analyzer():"),
  code("    global _instance"),
  code("    if _instance is None:"),
  code("        _instance = Analyzer()"),
  code("    return _instance"),
  p("Why? Creating these objects involves expensive operations: loading model clients, connecting to ChromaDB, setting up connection pools. We want to do this once at startup, then reuse. The singleton ensures this without passing objects around as function arguments."),

  h2("6.7 Session State Management — Why It's Centralized"),
  p("All st.session_state access goes through SessionManager in core/session.py, using the Keys class for all key names:"),
  code("Keys.CURRENT_SQL = 'current_sql'"),
  code("session.set_sql(sql_string)  # instead of st.session_state['current_sql'] = sql_string"),
  p("Why? Two reasons:"),
  bl("Typo prevention: If you misspell 'current_sql' as 'currrent_sql' in one place, you get a silent bug — you write to a different key and read from the original empty one. Using Keys.CURRENT_SQL, a typo causes a Python AttributeError which fails loudly."),
  bl("Single source of truth: Every session key and its purpose is documented in one place. No hunting through 20 files to find what 'pending_slicers' contains."),

  pageBreak(),
];

// ════════════════════════════════════════════════════════════════════════════
// DOC 7 — LOGGING AND BENCHMARKING
// ════════════════════════════════════════════════════════════════════════════

const doc7 = [
  h1("Document 7 — Logging and Benchmarking: How We Monitor the System"),
  p("This project generates structured logs for every operation. Understanding what each log file contains and how to read it is essential for debugging, performance analysis, and demonstrating the system's internals."),

  h2("7.1 Why Loguru Instead of Python's Built-in logging"),
  p("Python has a built-in logging module. We use loguru instead because:"),
  bl("Zero configuration: one import, immediately writes formatted logs with timestamps and line numbers"),
  bl("Automatic file rotation: log files are capped at 10MB and keep last 5 files automatically"),
  bl("Colored console output: different log levels appear in different colors, making development easier"),
  bl("Log binding: a logger can be 'bound' to a specific context — all messages from that logger include metadata"),
  p("The built-in logging module requires 15+ lines of setup code to achieve what loguru does in 5."),

  h2("7.2 The Five Log Files"),
  mkTable(
    ["File", "Contents", "Used For"],
    [
      ["app.log", "App startup, session creation, pipeline phase transitions", "General debugging, user activity"],
      ["model.log", "Every AI call: model name, prompt token count, latency, response preview", "AI performance analysis, prompt debugging"],
      ["db.log", "Every SQL query: preview, rows returned, execution time", "Database performance, SQL debugging"],
      ["charts.log", "Chart type, columns used, render time", "Chart selection debugging"],
      ["benchmark.log", "Step-by-step timing table for every user query", "Performance optimization, demo timing"],
    ]
  ),

  h2("7.3 Reading the Benchmark Log"),
  p("The most useful log for demos is benchmark.log. After each complete query, it writes a formatted timing table:"),
  code("════════════════════════════════════════════════════"),
  code('QUERY : "Show bounce rate by branch"'),
  code("  ├── intent_analysis             : 548815ms  ✓"),
  code("  ├── clarifying_questions        : 287776ms  ✓"),
  code("  ├── sql_generation              : 345185ms  ✓"),
  code("  ├── db_fetch                    :     36ms  ✓"),
  code("  ├── chart_decision              :  58907ms  ✓"),
  code("  └── chart_render               :    312ms  ✓"),
  code("════════════════════════════════════════════════════"),
  code("  TOTAL                         : 1241031ms"),
  code("════════════════════════════════════════════════════"),
  p("From this you can immediately see: the AI model calls (intent, clarifying, SQL, chart decision) account for almost all the time. The actual database query took only 36ms. Chart rendering took 312ms. The bottleneck is AI inference speed, not the database."),

  h2("7.4 The QueryBenchmark Class"),
  p("QueryBenchmark in utils/benchmark.py tracks timing across multiple pipeline steps for a single query. It is used like this:"),
  code("qb = QueryBenchmark('Show bounce rate by branch')"),
  code("qb.start('intent_analysis')"),
  code("# ... run intent analysis ..."),
  code("qb.end('intent_analysis')"),
  code("qb.start('sql_generation')"),
  code("# ... generate SQL ..."),
  code("qb.end('sql_generation')"),
  code("qb.report()  # writes the table to benchmark.log"),
  p("The report() method is also called to_dict() for the debug panel in the UI — when 'Show Timings' is checked, the user sees the step timings as metric cards."),

  h2("7.5 The benchmark() Context Manager"),
  p("For individual operations inside a single step, the benchmark() context manager measures a block of code:"),
  code("with benchmark('db_execute'):"),
  code("    result = pd.read_sql(sql, conn)"),
  p("This writes [START] and [END] entries to benchmark.log with the duration. Every database call, ChromaDB search, and chart render is wrapped in this so we have granular timing data."),

  h2("7.6 How to Debug Using the Logs"),
  p("When something goes wrong, read the logs in this order:"),
  bl("app.log: Find the session ID and see what phase the pipeline reached"),
  bl("model.log: Find the model call that failed. Look for TIMEOUT, HTTP ERROR, or JSON parse failed"),
  bl("db.log: Find the SQL query that was executed. Look for column name errors in the SQL preview"),
  bl("benchmark.log: Find the step that took suspiciously long or shows ERROR status"),
  p("The logs are structured so you can grep for specific patterns:"),
  code("grep 'FAILED' D:\\Loan_Dashboard\\Logs\\model.log"),
  code("grep 'intent_analysis' D:\\Loan_Dashboard\\Logs\\benchmark.log"),

  pageBreak(),
];

// ════════════════════════════════════════════════════════════════════════════
// DOC 8 — DECISIONS, TRADEOFFS, AND ALTERNATIVES REJECTED
// ════════════════════════════════════════════════════════════════════════════

const doc8 = [
  h1("Document 8 — Decisions, Tradeoffs, and Alternatives Rejected"),
  p("Every architectural decision involves tradeoffs. This document explains the key decisions, why we made them, what we gave up, and what we gained. Being able to articulate these tradeoffs shows deep understanding of the project."),

  h2("8.1 Rule-First Chart Selection vs Model-First"),
  h3("The decision"),
  p("Chart type is decided by rules (METRIC_CHART_RULES dictionary + DataFrame shape analysis) first. Qwen 7B only validates/refines the rule-based decision."),
  h3("The alternative we rejected"),
  p("Model-first: ask Qwen 7B to decide the chart type with no rules, then validate."),
  h3("Why we rejected it"),
  p("The screenshots showed Qwen 7B picking scatter charts for 134-branch categorical data, and line charts for branch rankings. The model was making decisions based on surface patterns in the question text ('by branch' → 'scatter because there are two axis-like things') without understanding what makes a chart readable."),
  h3("What we gave up"),
  p("Flexibility — a purely rule-based system cannot discover a chart type we did not anticipate. The model might know a better chart type for unusual data shapes."),
  h3("What we gained"),
  p("Reliability — for the 14 known metric types, we always get the correct chart. For unknown query shapes, the fallback rules still produce something sensible."),

  h2("8.2 Mock Data vs Live Data for Development"),
  h3("The decision"),
  p("USE_MOCK_DATA=true in .env bypasses PostgreSQL entirely and returns synthetic data from mock.py. The mock generates 89,255 rows with realistic distributions and correlations."),
  h3("The alternative"),
  p("Always use real data. Never have a mock mode."),
  h3("Why we chose mock mode"),
  p("Development requires running the same query 50 times while debugging. Each query takes 15-60 seconds just for the AI steps, plus 25-100ms for the database. With mock mode, we can iterate on the chart rendering code without waiting for the full AI pipeline on every test."),
  p("For demos without database access (presenting at a client site, testing on a laptop without Docker), mock mode means the system still works and shows realistic data."),
  h3("The risk"),
  p("Mock data never perfectly replicates real data distributions. A chart that looks good with mock data might have edge cases with real data (null values in unexpected columns, values outside expected ranges)."),

  h2("8.3 Two-Phase Pipeline vs Single-Phase"),
  h3("The decision"),
  p("The pipeline is split into Phase 1 (question understanding) and Phase 2 (data fetching) with a Streamlit state machine managing transitions."),
  h3("What we tried first"),
  p("A single pipeline function that showed filter widgets mid-execution and then continued after the user clicked Generate Dashboard."),
  h3("Why it failed"),
  p("Streamlit reruns the entire script on every button click. When the user clicked 'Generate Dashboard', Streamlit ran the whole script again from the top. The single-function pipeline had no way to 'remember' it was in the middle of execution — it restarted Phase 1 completely, showing the question input again instead of continuing to Phase 2."),
  h3("The fix"),
  p("Every output from Phase 1 is saved to session state before Streamlit reruns. Phase 2 reads from session state instead of re-running Phase 1. The pipeline_phase variable tracks which phase we are in so the correct UI is shown on each rerun."),

  h2("8.4 Plotly vs Matplotlib for Charts"),
  h3("The decision"),
  p("All charts use Plotly (interactive). Matplotlib is kept only for PDF export."),
  h3("Why we switched"),
  p("The original Seaborn/Matplotlib charts (visible in screenshots 1-3) had problems: the 134-branch line chart was an unreadable zigzag. There was no way to hover to see exact values. The chart size was controlled by matplotlib figure settings and did not resize properly in Streamlit."),
  h3("What Plotly adds"),
  p("Hover tooltips show exact values for any data point. Zoom and pan are built in. The chart resizes automatically to the container width. Users can download the chart as PNG directly from the chart toolbar. All of this comes for free — no additional code."),
  h3("The tradeoff"),
  p("Plotly charts are rendered as JavaScript in the browser, not as server-side images. This means they cannot be embedded in PDF reports as interactive charts — we must render them to static PNG first using kaleido for PDF export."),

  h2("8.5 Local Models vs Cloud APIs"),
  h3("The decision"),
  p("All AI inference uses local Ollama models. No calls to OpenAI, Anthropic, Google, or any cloud AI service."),
  h3("The business reason"),
  p("Loan collection data contains customer names, payment histories, and financial details. Regulatory and contractual requirements prohibit this data from being sent to third-party cloud services. Even if the cloud API is only seeing the SQL query (not the raw data), the query itself contains field names and values that reveal business logic."),
  h3("The performance cost"),
  p("Without a dedicated GPU, local models are 10-100x slower than cloud APIs. GPT-4o responds in 2-5 seconds. Qwen Coder 14B on CPU takes 5-10 minutes. This is the biggest limitation of the current system."),
  h3("The solution path"),
  p("A dedicated GPU (RTX 3090 or 4090, ~$600-1200) reduces model response time to 15-40 seconds per call, making the total pipeline 2-3 minutes — acceptable for business use."),

  h2("8.6 ChromaDB vs Sending All Context Every Time"),
  h3("The decision"),
  p("Store context in ChromaDB and retrieve ~1,400 relevant tokens per query instead of sending all 15,000 tokens."),
  h3("Why it matters"),
  p("At temperature 0.1, Qwen Coder 14B processes approximately 2,000 tokens per second on a modern GPU. Sending 15,000 tokens takes 7.5 seconds just for input processing. Sending 1,400 tokens takes 0.7 seconds. Across 5 model calls per query, this saves 35 seconds."),
  p("Additionally, research consistently shows that large language models pay less attention to information in the middle of very long contexts (the 'lost in the middle' problem). Shorter, focused context produces more accurate outputs."),

  pageBreak(),
];

// ════════════════════════════════════════════════════════════════════════════
// DOC 9 — COMPLETE DEMO GUIDE AND Q&A PREPARATION
// ════════════════════════════════════════════════════════════════════════════

const doc9 = [
  h1("Document 9 — Complete Demo Guide and Q&A Preparation"),
  p("This document prepares you to present this project to any audience — technical developers, business stakeholders, senior management, or external clients. It includes the exact demo script, every question you might be asked, and how to handle technical difficulties gracefully."),

  h2("9.1 Understanding Your Audience"),
  p("Different people care about different aspects of this project:"),
  mkTable(
    ["Audience", "What They Care About", "How to Frame It"],
    [
      ["Collections Manager", "Can I get answers without calling the data team?", "Focus on the question-to-chart flow. Show real business metrics."],
      ["IT/Technical Team", "Is it secure? How does it connect to the database? What models?", "Focus on local execution, PostgreSQL connection, no cloud APIs."],
      ["Senior Management", "What business value does this create? How much does it cost?", "Focus on analyst time saved, self-service analytics, zero API cost."],
      ["Data Scientist", "How accurate is the SQL? How does the model know the schema?", "Focus on ChromaDB, context engineering, rule-based fallbacks."],
      ["External Client", "Can we buy/deploy this? Is it customizable?", "Focus on the JSON-based configuration that makes it data-agnostic."],
    ]
  ),

  h2("9.2 The Exact Demo Script"),
  p("Follow this sequence. Each step has a specific purpose."),
  h3("Before you start (setup checklist)"),
  bl("Docker running: docker ps shows the loan-db container"),
  bl("Ollama running: ollama serve in a terminal, both models loaded"),
  bl("App running: streamlit run app.py, browser open to localhost:8501"),
  bl("Sidebar shows: Database Connected, Ollama Running, Coder ✅, Fast ✅, Context Loaded"),
  bl("Row count shows: 89,255 rows in loan_dashboard"),
  spacer(),
  h3("Step 1: Show the system status (30 seconds)"),
  p("Point to the sidebar. Say: 'Everything you see here runs on this machine. The database is local Docker PostgreSQL. The AI models are local Ollama. No data leaves this network.'"),
  h3("Step 2: Click an example question (1 minute)"),
  p("Click 'Show bounce rate by branch'. Say: 'I am asking a business question in plain English. No SQL, no filter configuration, no chart type selection.'"),
  h3("Step 3: Explain the processing steps as they appear"),
  p("While the status bars run: 'The system is doing three things: first, understanding what I asked and identifying it as a Bounce Rate question. Second, figuring out which filters make sense. Third, generating the database query automatically.'"),
  h3("Step 4: Answer the filter questions"),
  p("Leave everything as All. Say: 'It is asking me to narrow down — which region, which portfolio. I will leave them all to see the full picture.' Click Generate Dashboard."),
  h3("Step 5: Show the result"),
  p("When the horizontal bar chart appears: 'This shows all branches ranked by bounce rate. I can hover over any bar to see the exact value.' Hover over a bar to demonstrate."),
  p("Click the Insights tab: 'The AI has read the data and written 5 business observations. INDORE has the highest bounce count. The top 3 branches account for X% of all bounces. These are the kinds of observations a senior analyst would manually write.'"),
  h3("Step 6: Show the Data and SQL tabs"),
  p("Click Data: 'Full data available, downloadable as CSV or Excel.' Click SQL: 'This is the exact query that ran. Completely transparent — you can see how the system is getting the answer.'"),
  h3("Step 7: Show chart type switching"),
  p("Click the dropdown below the chart, change from horizontal_bar to heatmap. Click Apply. Say: 'The chart type is not locked in. You can switch to any of 8 chart types with one click.'"),
  h3("Step 8: Try a different question"),
  p("Click New Query. Click 'Show bucket movement matrix'. This shows a heatmap automatically. Say: 'Different question, different metric, completely different chart type — the system adapts automatically.'"),

  h2("9.3 Every Question You Will Be Asked"),
  h3("'Why not just use Power BI directly?'"),
  p("Power BI requires the analyst to know which columns to use, how to configure the filter panel, and which chart type is appropriate. A collections manager who is not a data analyst must either learn these skills or wait for an analyst. This system removes both requirements — the manager types what they want in plain English and gets an answer in 2-3 minutes."),
  h3("'Is the data secure?'"),
  p("Completely. The AI models (Qwen Coder 14B and Qwen 7B) run locally via Ollama — they never make network calls to any external server. The database is a local Docker container. The Streamlit app is a local web server. Nothing in this system has an internet connection during normal operation."),
  h3("'How accurate is the SQL it generates?'"),
  p("We have two layers of safety. First, the system has pre-built SQL templates for all 14 most common metrics. Qwen adapts these templates rather than writing from scratch, dramatically reducing errors. Second, after Qwen generates SQL, it is validated: must be a SELECT statement, no modification keywords allowed, must reference the correct table. If validation fails, the rule-based fallback generates correct SQL without calling the model. In testing, the SQL is correct 95%+ of the time for the known metric types."),
  h3("'What if it picks the wrong chart?'"),
  p("The user has a chart type dropdown immediately below the chart. They can switch to any of 8 chart types with one click and the chart re-renders instantly. The AI's suggestion is a starting point, not a lock-in."),
  h3("'How fast is it?'"),
  p("Currently running on CPU without a dedicated GPU, each AI call takes 5-10 minutes. Total query time is 20-30 minutes. With a dedicated GPU (RTX 3090 or 4090, approximately ₹80,000-₹1,20,000), each AI call drops to 15-40 seconds and total query time is 2-3 minutes. The database query itself takes only 25-100ms regardless of hardware."),
  h3("'Can we add our own data or more questions?'"),
  p("Yes. All the domain knowledge is stored in 5 JSON files in the Context folder. To add new column definitions, you update 01_schema_context.json. To add new SQL patterns for different metrics, you update 05_sql_patterns.json. No Python code changes are needed. The system will automatically pick up the new knowledge when ChromaDB reloads."),
  h3("'What happens if the AI gives a wrong answer?'"),
  p("The SQL tab shows exactly what query ran. The user can see the precise question the AI answered. If the SQL looks wrong, they can copy it, fix it manually, and report the issue. The model.log file records every AI call with the full prompt and response, making debugging straightforward."),
  h3("'Can this work with a different dataset?'"),
  p("Yes — the system is data-agnostic. The connection point between the business domain and the AI is the 5 context JSON files. To deploy this for a different dataset (home loans, MSME loans, a different company's portfolio), you would update the JSON files with the new column definitions and reindex ChromaDB. The Python code, the models, the Streamlit app — none of that changes."),
  h3("'Why does it take so long on CPU?'"),
  p("Running a 14-billion parameter model on CPU is like computing with 14 billion calculations for every token generated. A modern CPU can do approximately 100-200 tokens per second for this model size. A GPU (designed for massively parallel computation) does 2,000-5,000 tokens per second. This is not a software limitation — it is a fundamental hardware difference."),
  h3("What does 14B mean in 'Qwen 14B'?"),
  p("14 billion parameters. Parameters are the numerical weights inside the neural network — the learned 'knowledge' that determines how the model responds to inputs. More parameters means the model can capture more complex patterns and reasoning, but requires more memory and compute. 14B requires approximately 10-14GB of RAM/VRAM depending on quantization."),
  h3("'What is quantization?'"),
  p("Quantization is a compression technique. A full-precision model stores each parameter as a 32-bit floating point number. Q4 quantization stores each parameter as a 4-bit number. This reduces memory by 8x with a small loss in output quality. Qwen 14B in Q4 quantization fits in about 9GB of VRAM, making it practical on a single RTX 3090 GPU."),

  h2("9.4 Handling Technical Difficulties During Demo"),
  h3("If the model times out"),
  p("Say: 'The model call timed out — this happens occasionally on CPU hardware. The system will retry automatically.' While it retries, explain the benchmark log showing what the system is doing. If it fails twice, switch to mock mode."),
  h3("If the SQL generates an error"),
  p("Say: 'Let me show you the fallback system.' Click New Query and try the same question — the rule-based SQL generator will produce correct SQL without the model. Explain: 'We have two layers — AI-generated SQL and a rule-based backup. The backup covers all the standard metrics.'"),
  h3("If the chart looks wrong"),
  p("Use the chart type dropdown to switch to a different visualization. Say: 'The AI suggested this chart type based on the data shape. Let me show you another view.' This demonstrates the flexibility rather than hiding the problem."),

  h2("9.5 Glossary — Every Technical Term Explained"),
  mkTable(
    ["Term", "Plain English Explanation"],
    [
      ["LLM", "Large Language Model — an AI that processes and generates text, like GPT-4 or Qwen"],
      ["Parameters (in a model)", "The numerical weights inside an AI model that define its 'knowledge'. More = smarter but slower."],
      ["Quantization", "Compressing a model by reducing numerical precision — makes it smaller and faster at small quality cost"],
      ["Ollama", "Software that runs AI models on your own computer — like having ChatGPT but private and local"],
      ["Token", "A chunk of text (roughly a word or word-piece). Models process and generate text in tokens."],
      ["Temperature", "Controls randomness in model output. 0 = always same answer. 1 = more creative/random."],
      ["Vector / Embedding", "A list of numbers that represents the meaning of text. Similar meanings → similar numbers."],
      ["ChromaDB", "A database that stores vectors and finds the most similar ones to a query — used for semantic search"],
      ["Context window", "The maximum amount of text an AI model can process at once. Qwen 14B = 128K tokens."],
      ["Streamlit", "Python library that creates web apps — write Python, get a webpage automatically"],
      ["Session state", "Memory that survives Streamlit's script reruns — where we store pipeline results between steps"],
      ["PostgreSQL", "A relational database — stores data in tables with rows and columns, uses SQL for queries"],
      ["Docker", "Packages software into containers — easy to start, stop, and move between computers"],
      ["SQLAlchemy", "Python library that manages database connections efficiently using a connection pool"],
      ["Connection pool", "Pre-opened database connections that queries can reuse instead of opening new ones each time"],
      ["Plotly", "Python library for interactive charts — hover, zoom, pan built-in"],
      ["DAX", "Data Analysis Expressions — query language for Power BI (we switched from this to SQL)"],
      ["snake_case", "Naming convention: all lowercase with underscores — op_bucket, cust_id, bounce_status"],
      ["DPD", "Days Past Due — how many days overdue a loan payment is"],
      ["NPA", "Non-Performing Asset — a loan where the borrower has not paid for 90+ days"],
      ["NACH", "National Automated Clearing House — automatic bank debit for recurring payments like EMIs"],
      ["EMI", "Equated Monthly Installment — the fixed monthly payment amount for a loan"],
      ["NBFC", "Non-Banking Financial Company — a company that provides financial services but is not a bank"],
      ["Cosine similarity", "A mathematical measure of how similar two vectors are — used by ChromaDB to find relevant context"],
    ]
  ),

  pageBreak(),
];

// ════════════════════════════════════════════════════════════════════════════
// BUILD DOCUMENT
// ════════════════════════════════════════════════════════════════════════════

const allContent = [
  ...titlePage,
  ...doc1, ...doc2, ...doc3,
  ...doc4, ...doc5, ...doc6,
  ...doc7, ...doc8, ...doc9,
];

const document = new Document({
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{
        level: 0,
        format: LevelFormat.BULLET,
        text: "•",
        alignment: AlignmentType.LEFT,
        style: {
          paragraph: {
            indent: { left: 720, hanging: 360 },
            spacing: { after: 100 },
          },
        },
      }],
    }],
  },
  styles: {
    default: {
      document: {
        run: { font: "Arial", size: 22, color: "000000" },
      },
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1",
        basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "Arial", color: "1F4E79" },
        paragraph: { spacing: { before: 480, after: 240 }, outlineLevel: 0 },
      },
      {
        id: "Heading2", name: "Heading 2",
        basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: "2E75B6" },
        paragraph: { spacing: { before: 320, after: 160 }, outlineLevel: 1 },
      },
      {
        id: "Heading3", name: "Heading 3",
        basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: "006064" },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2 },
      },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 },
      },
    },
    children: allContent,
  }],
});

Packer.toBuffer(document).then(buffer => {
  fs.writeFileSync('/home/claude/docs9/Loan_Analytics_Complete_Guide.docx', buffer);
  console.log('✅ Done — 9 documents generated');
}).catch(e => {
  console.error('❌ Error:', e.message);
  process.exit(1);
});