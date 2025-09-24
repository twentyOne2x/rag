from enum import Enum


# SYSTEM_MESSAGE = """
# You are an expert in Maximal Extractable Value (MEV) that answers questions using the tools at your disposal.
# These tools have information regarding MEV research including academic papers, articles, diarized transcripts from conversations registered on talks at conferences or podcasts.
# Here are some guidelines that you must follow:
# * For any user message that is not related to MEV, blockchain, or mechanism design, respectfully decline to respond and suggest that the user ask a relevant question.
# * If your tools are unable to find an answer, you should say that you haven't found an answer.
#
# Now answer the following question:
# {question}
# """.strip()

#  that answers questions using the query tools at your disposal.

# If the user requested sources or content, return the sources regardless of response worded by the query tool.

# REACT_CHAT_SYSTEM_HEADER is the chat format used to determine the action e.g. if the query tool should be used or not.
# It is tweaked from the base one.

# You are designed to help with a variety of tasks, from answering questions \
# to providing summaries to providing references and sources about the requested content.


# You are responsible for using
# the tool in any sequence you deem appropriate to complete the task at hand.
# This may require breaking the task into subtasks and using different tools
# to complete each subtask.


LLM_TEMPERATURE = 0
NUMBER_OF_CHUNKS_TO_RETRIEVE = 10
TEXT_SPLITTER_CHUNK_SIZE = 700
TEXT_SPLITTER_CHUNK_OVERLAP_PERCENTAGE = 10

'''
valid OpenAI model name in: gpt-4, gpt-4-32k, gpt-4-0613, gpt-4-32k-0613, gpt-4-0314, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-16k, gpt-3.5-turbo-0613, 
gpt-3.5-turbo-16k-0613, gpt-3.5-turbo-0301, text-davinci-003, text-davinci-002, gpt-3.5-turbo-instruct, 
text-ada-001, text-babbage-001, text-curie-001, ada, babbage, curie, davinci, gpt-35-turbo-16k, gpt-35-turbo, gpt-3.5-turbo-0125
'''
OPENAI_INFERENCE_MODELS = ["gpt-4o-mini", "gpt-3.5-turbo-0125", "gpt-4", "gpt-4-32k", "gpt-4-0613", "gpt-4-32k-0613", "gpt-4-0314", "gpt-4-32k-0314", "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0613",
"gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0301", "text-davinci-003", "text-davinci-002", "gpt-3.5-turbo-instruct", "gpt-35-turbo-16k", "gpt-35-turbo", "gpt-4-1106-preview", "gpt-4-turbo", "gpt-4-turbo-1106"]


OPENAI_MODEL_NAME = "gpt-3.5-turbo"  #"gpt-3.5-turbo-16k-0613"  # "gpt-3.5-turbo-0613"  # "gpt-4" # "gpt-3.5-turbo-0613"  # "gpt-4-0613"  # "gpt-3.5-turbo-0613"
INPUT_QUERIES = [
    "How does Cupsey make money?",
    "What is a DAT as per Kyle Samani?",
    "What's Pump.fun long term vision?"
]
INPUT_QUERIES_test = [
    # — Beginner: short, friendly, fundamentals (1–16)


    "What is Solana in simple terms?",  # videos: Solana Seeker - The Web3 Mobile Evolution; A Better Internet, Built on Solana; Spotlight: Solana's Consensus
    "What is SOL used for on the Solana network?",  # videos: The State of the USDC Economy Is Strong; Solana Seeker Keynote (Emmett Hollyer); Solana Stories: Fast, Cheap, and Reliable
    "What is a memecoin and why are they popular on Solana?",  # videos: Make 100x Money With Memecoins TODAY!; Did Memecoins SAVE Crypto??; The TRUTH About Crypto Trading In 2024... w/ Ubermendes
    "What is Pump.fun and how does it work at a high level?",  # videos: How Pumpfun ACTUALLY works | Founder Explains for BEGINNERS; How Memecoins Will Get You RICH | Pump Fun Founder's Full Insight; The Bull Thesis for Pumpfun
    "What does “trench” mean in crypto launches and why do people talk about it?",  # videos: Letterbomb: The Dark Side of Trenching; Gwart & Robert Chang: Trenching...; The TRUTH About Crypto Trading In 2024... w/ Ubermendes
    "How do I set up a Solana wallet and avoid common scams?",  # videos: How to Make Your Transactions Faster on #Solana; Building web3 Login Your Grandma Can Use; Developer Tutorial: How to Verify Your Solana Programs
    "What’s the difference between a DEX and a CEX on Solana?",  # videos: The OG of Solana DeFi | Orca; Coinbase & DeFi; DRIFT: Building a Composite CEX-DEX Experience
    "What are Raydium and Orca, and what can I do with them?",  # videos: The OG of Solana DeFi | Orca; Does DEX Liquidity Need a Defense Layer?; Adapting DEX Aggregation to Solana
    "What is slippage and why does it matter when trading on Solana?",  # videos: How to SELL Your MEMECOINS for MAX PROFIT!! w/ Ansem; On-Chain Liquidity: The Past, Present and Future
    "What is an airdrop, and how do people qualify on Solana?",  # videos: The Infinite-LST Future w/ FP Lee; Solana Incubator Cohort 2 Demo Day; Jupiter (Breakpoint 2024)
    "What are Solana Actions and Blinks, and why do people use them?",  # videos: Solana Actions and Blinks; What Are You Building? with Dialect; In the Blink of an Eye (Chris Osborn)
    "What is Jito and how can it improve my transaction experience?",  # videos: Solana Validator Education - Jito Relayer Setup; The State of Solana MEV; Jito StakeNet: A Protocol for Timeless LSTs
    "What is an AMM, explained like I’m new to DeFi?",  # videos: The Phoenix Rises: Why Active Liquidity Enables Sustainable Markets; Widening the Design Space of AMMs with Solana; What Are You Building? with Orca
    "What is a bonding curve on Pump.fun?",  # videos: How Pumpfun ACTUALLY works | Founder Explains; The Bull Thesis for Pumpfun; Legion: The Return of ICOs
    "What is a DAT (Digital Asset Token)?",  # videos: Kyle Samani - $1.65B Solana DAT; Joseph Onorati: Solana DATs; Internet Capital Markets (Akshay BD)
    "What are oracles on Solana, and what does Pyth do?",  # videos: Pyth Network: Supercharged DeFi Infrastructure; Validated | How Pyth Is Changing the Oracle Game; Walking Through the Pyth Whitepaper

    # — Intermediate: Solana, Pump.fun, trenches, DATs (17–35)
    "How does a Pump.fun launch flow from mint to ‘graduation’ onto a DEX?",  # videos: How Pumpfun ACTUALLY works | Founder Explains; The OG of Solana DeFi | Orca; Adapting DEX Aggregation to Solana
    "How can I spot red flags and avoid rugs in trench-style launches?",  # videos: Letterbomb: The Dark Side of Trenching; Critical Security Considerations for Web3 Builders; Safe Solana Stack Smashing by OtterSec
    "How do priority fees work on Solana, and when should I use them?",  # videos: Solana Changelog Apr 2 - priority fees; Solana Changelog May 1 - priority fees; Stake Weighted QoS
    "What is Firedancer and why is everyone excited about it?",  # videos: Firedancer w/ Kevin Bowers; Fast Forward From Frankendancer to Firedancer; Spotlight: Firedancer v0 Architecture
    "How does MEV work on Solana at a high level (no math)?",  # videos: The State of Solana MEV; WTF Is MEV and Why Should We Care?; Solana Validator Education - Jito-Solana Concepts
    "What is Token-2022 and how do transfer hooks help protect users?",  # videos: Token Extensions on Solana series; Code With the Transfer Hooks Token Extension; Token 2022 in 2023 will define 2024
    "What’s impermanent loss and how does providing LP on Raydium actually work?",  # videos: The Phoenix Rises: Why Active Liquidity...; Does DEX Liquidity Need a Defense Layer?; What Are You Building? with Orca
    "What are LSTs like JitoSOL or Marinade, and why would I hold them?",  # videos: Water from a Stone: Liquid Staking on Solana; Jito StakeNet; Marinade (Breakpoint 2024 Product Keynote)
    "How can wallets simulate a trade to warn me about bad outcomes?",  # videos: What Are You Building? with Dialect; Safe Solana Stack Smashing by OtterSec; Real-Time Security in Solana Ecosystem
    "What’s the difference between AMMs and order books like Phoenix/Ellipsis?",  # videos: The Phoenix Rises; Atlas: Verifiable Finance At Scale; Adapting DEX Aggregation to Solana
    "What is a DAT vs an RWA, and how are they different from traditional stocks?",  # videos: Kyle Samani - $1.65B Solana DAT; Tokenization of Capital Markets (Ondo); BP 2024: Société Générale Forge
    "How could creators use Solana Attestation Service (SAS) to build trust?",  # videos: Solana Attestation Service: Attest to Anything; Ship or Die 2025: SAS talk; What Are You Building? with Dynamic
    "What are on-chain allowlists/KYC gates and do they ruin UX?",  # videos: Token Extensions on Solana - Regulatory-friendly Finance; Compliant Onchain Products (Exo); Crypto Regulation: Principles-Based Approach
    "How do perps on Drift or Zeta work for someone coming from spot?",  # videos: DRIFT: Building a Composite CEX-DEX Experience; How Zeta Markets’ L2 Makes a DEX Feel Like a CEX; Scale or Die: The road to Decentralized Nasdaq
    "How can Blinks/Actions make buying a coin or DAT one-tap from social media?",  # videos: Solana Actions and Blinks; Build Omnichain Programs; In the Blink of an Eye
    "How do oracles get manipulated and what basic defenses exist?",  # videos: Pyth: Past, Present, and Future; Real-Time Security in Solana Ecosystem; Security and Risk Monitoring for Solana
    "What simple steps should creators take to launch responsibly (vesting, locks, multisig)?",  # videos: Squads Labs: Accelerating the Onchain Economy; Forgd: Tokenomics, Liquidity, and Beyond; Safe Solana Stack Smashing by OtterSec
    "What telemetry from RPCs/validators helps retail get better fills on Solana?",  # videos: WTF RPC?; Solana RPC 2.0 Roundtable; High Performance Networking: DoubleZero Enables IBRL
    "What’s the safest way to bridge assets into Solana for new users?",  # videos: Bridge: Make Money Move; Wormhole Product Keynotes; Debridge: DeFi Doesn't Wait
    "What education should wallets show to explain ‘attention risk’ in trenches?",  # videos: Letterbomb: The Dark Side of Trenching; Consumer Apps will Eat the World; Resetting Consumer Expectations

    # — Advanced: design, market structure, policy, research (36–50)
    "How could a Solana-native DAT issuance work end-to-end (KYC, bookbuild, allocation, settlement)?",  # videos: Kyle Samani - $1.65B Solana DAT; Tokenization of Capital Markets (Ondo); Internet Capital Markets (Akshay BD)
    "What are the compliance boundaries for creator/streamer tokens on Solana?",  # videos: Crypto Regulation: Principles-Based Approach; Compliant Onchain Products; The Little JPG that Laundered Money
    "How do Pump.fun incentives (fees, graduation rules) align creators, LPs, and traders?",  # videos: How Pumpfun ACTUALLY works | Founder Explains; Mike Dudas: Pump.Fun seed investment; The Bull Thesis for Pumpfun
    "Could batch auctions or OFAs make meme launches fairer on Solana?",  # videos: The State of Solana MEV; The Phoenix Rises; Multi-dimensional Fee Markets (Tarun Chitra)
    "Would commit–reveal or encrypted mempools reduce sniping and sandwiches on Solana?",  # videos: Composable Privacy with Sandwiching; Encrypt or Die (Arcium); The State of Solana MEV
    "How does Solana MEV (Jito bundles/BAM) differ from Ethereum, and who benefits?",  # videos: The State of Solana MEV; Solana Validator Education - Jito; Why MEV Is Here to Stay (Breakpoint 2022)
    "What is Alpenglow and how might it change latency/bandwidth tradeoffs?",  # videos: Introducing Alpenglow - Solana’s New Consensus; Increase Bandwidth, Reduce Latency; Spotlight: Solana's Scheduler
    "How would multi-dimensional fee markets affect UX during peak trench activity?",  # videos: Multi-dimensional Fee Markets (Tarun Chitra); Solana Changelog - fees topics; Stake Weighted QoS
    "What’s the right architecture for a social trading app on Solana with copy-trading and circuit breakers?",  # videos: What Are You Building? with Ranger Finance; One App for Web3, AI, Messaging | Zo; Consumer Apps will Eat the World
    "Should Pump.fun offer DATs, and what legal/custody hurdles must be solved first?",  # videos: Kyle Samani - $1.65B Solana DAT; Compliant Onchain Products; Crypto Regulation: Principles-Based Approach
    "How do LSTs/LRTs interact with using DATs as collateral or for margining?",  # videos: Water from a Stone: Liquid Staking; Jito StakeNet; Atlas: Verifiable Finance At Scale
    "Could DePIN networks issue DAT-like cash-flow tokens on Solana?",  # videos: DePIN on Solana: Helium; DePIN on Solana: WeatherXM; DePIN on Solana: GEODNET
    "How could a Solana market-data business monetize without harming open access?",  # videos: Finance's Best-Kept Secret: Market Data Powers Everything (Pyth); Birdeye Product Keynote; Real-Time Security in Solana Ecosystem
    "Is payment-for-order-flow (PFOF) desirable on Solana DEX aggregators?",  # videos: Mantis: Powering Best Execution; Adapting DEX Aggregation to Solana; Phoenix/Ellipsis talks (Atlas/Verifiable Finance)
    "Are shared sequencers or SVM rollups useful for high-risk trenches, or is mainnet better?",  # videos: Eclipse: Ethereum's First SVM L2; SVM: Power of Solana Beyond the Blockchain; The road to Decentralized Nasdaq
    "What protections can wallets add to stop malicious transfer-hook tokens draining users?",  # videos: Token Extensions on Solana series; Safe Solana Stack Smashing by OtterSec; Real-Time Security in Solana Ecosystem
    "How should a risk dashboard track trench coins, perps exposure, and DAT holdings together?",  # videos: Atlas: Verifiable Finance At Scale; The State of Solana MEV; What Are You Building? with Birdeye
    "What roles should exist in a 2026 Solana MEV supply chain (searcher, builder, solver, proposer)?",  # videos: The State of Solana MEV; Solana Validator Education - Jito; Designing the MEV Transaction Supply Chain (talks set)
    "Should AI agents be allowed to launch and trade coins on Solana, and what guardrails are needed?",  # videos: Where AI Meets Web3; Crypto x AI: Investing Across The Stack; AI Agents Will Change Crypto (EigenCloud)
    "What does ‘Internet Capital Markets’ on Solana really mean between 2025–2026, and who wins?",  # videos: Internet Capital Markets (Akshay BD); The Future of Digital Assets; Kyle Samani: Solana Treasury / Internet Markets
]


# * Even if it seems like your tools won't be able to answer the question, you must still use them to find the most relevant information and insights. Not using them will appear as if you are not doing your job.
# * You may assume that the users financial questions are related to the documents they've selected.

# The tools at your disposal have access to the following SEC documents that the user has selected to discuss with you:
#     {doc_titles}
# The current date is: {curr_date}


class DOCUMENT_TYPES(Enum):
    YOUTUBE_VIDEO = "youtube_video"
    RESEARCH_PAPER = "research_paper"
    ARTICLE = "article"

