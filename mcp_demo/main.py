import os
import sqlite3
from pathlib import Path
from fastmcp import FastMCP
from fastmcp.exceptions import ResourceError


mcp = FastMCP("Customer Support")

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "customers.db")


def get_db_connection():
    """
    Get a connection to the database.
    """

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # This allows us to access columns by name
    return conn


def get_customer(customer_id: str) -> dict | None:
    """
    Get a customer from the database including their subscription information and all support tickets.

    Args:
        customer_id: The customer ID to retrieve

    Returns:
        Dictionary containing customer data, subscription info, and list of support tickets, or None if not found
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get customer and subscription data
    cursor.execute("""
        SELECT c.*, s.*
        FROM customers c
        LEFT JOIN subscriptions s ON c.id = s.customer_id
        WHERE c.id = ?
    """, (customer_id,))
    customer_row = cursor.fetchone()

    if customer_row is None:
        conn.close()
        return None

    customer_data = dict(customer_row)

    # Get all support tickets for this customer with contact information
    cursor.execute("""
        SELECT
            st.id as ticket_id,
            st.subject,
            st.description,
            st.status,
            st.priority,
            st.created_date,
            st.resolved_date,
            co.name as contact_name,
            co.title as contact_title,
            co.email as contact_email,
            co.primary_contact
        FROM support_tickets st
        LEFT JOIN contacts co ON st.contact_id = co.id
        WHERE st.customer_id = ?
        ORDER BY st.created_date DESC, st.priority DESC
    """, (customer_id,))

    tickets = cursor.fetchall()
    conn.close()

    # Add tickets to customer data
    customer_data['tickets'] = [dict(ticket) for ticket in tickets]

    return customer_data




@mcp.tool()
def get_support_tickets(customer_id: str, timeframe: str = "30days") -> str:
    """
    Get and format support tickets for a specific customer within a given timeframe.
    Uses the existing get_customer function to fetch customer details and presents
    all tickets in a human-readable format.

    Args:
        customer_id: The customer ID to fetch tickets for (required)
        timeframe: Time period to filter tickets (e.g., "30days", "7days", "90days", "1year")

    Returns:
        Human-readable formatted string with customer info and their support tickets
    """
    from datetime import datetime, timedelta
    import re

    # Parse timeframe string (e.g., "30days", "7days", "1year")
    timeframe_match = re.match(r'(\d+)(days?|weeks?|months?|years?)', timeframe.lower())
    if not timeframe_match:
        return f"Invalid timeframe format: {timeframe}. Use format like '30days', '7days', '1year', etc."

    amount, unit = timeframe_match.groups()
    amount = int(amount)

    # Calculate the date threshold
    today = datetime.now()
    if unit.startswith('day'):
        threshold_date = today - timedelta(days=amount)
    elif unit.startswith('week'):
        threshold_date = today - timedelta(weeks=amount)
    elif unit.startswith('month'):
        threshold_date = today - timedelta(days=amount * 30)  # Approximate
    elif unit.startswith('year'):
        threshold_date = today - timedelta(days=amount * 365)  # Approximate
    else:
        return f"Unsupported time unit: {unit}"

    threshold_date_str = threshold_date.strftime('%Y-%m-%d')

    # Get customer details with all tickets using existing function
    customer_data = get_customer(customer_id)
    if not customer_data:
        return f"Customer with ID '{customer_id}' not found."

    # Filter tickets by timeframe from the customer data
    all_tickets = customer_data.get('tickets', [])
    tickets = [
        ticket for ticket in all_tickets
        if ticket.get('created_date', '') >= threshold_date_str
    ]

    # Format the output
    priority_emojis = {'Critical': '🔴', 'High': '🟠', 'Medium': '🟡', 'Low': '🟢'}
    status_emojis = {'Open': '📂', 'In Progress': '⚙️', 'Resolved': '✅', 'Closed': '📁'}

    # Build the formatted response
    result = f"""
╭─────────────────────────────────────────────────────────────────╮
│ CUSTOMER SUPPORT TICKETS REPORT                                │
╰─────────────────────────────────────────────────────────────────╯

🏢 CUSTOMER INFORMATION
   Company: {customer_data.get('name', 'N/A')} ({customer_id})
   Industry: {customer_data.get('industry', 'N/A')}
   Size: {customer_data.get('size', 'N/A')}
   Country: {customer_data.get('country', 'N/A')}
   Customer Since: {customer_data.get('created_date', 'N/A')}

💼 SUBSCRIPTION DETAILS
   Plan: {customer_data.get('plan', 'N/A')}
   Seats: {customer_data.get('seats', 'N/A')}
   Monthly Value: ${customer_data.get('monthly_value', 0):.2f}
   Status: {customer_data.get('status', 'N/A')}
   Start Date: {customer_data.get('start_date', 'N/A')}
   Renewal Date: {customer_data.get('renewal_date', 'N/A')}

📊 TICKETS SUMMARY (Last {timeframe})
   Total Tickets: {len(tickets)}
   Date Range: {threshold_date_str} to {today.strftime('%Y-%m-%d')}

{'─' * 65}
"""

    if not tickets:
        result += "\n🎉 No support tickets found in the specified timeframe!\n"
        return result

    # Count tickets by status and priority
    status_counts = {}
    priority_counts = {}

    for ticket in tickets:
        status = ticket.get('status') or 'Unknown'
        priority = ticket.get('priority') or 'Unknown'
        status_counts[status] = status_counts.get(status, 0) + 1
        priority_counts[priority] = priority_counts.get(priority, 0) + 1

    # Add status breakdown
    result += "\n� STATUS BREAKDOWN\n"
    for status, count in status_counts.items():
        emoji = status_emojis.get(status, '❓')
        result += f"   {emoji} {status}: {count}\n"

    # Add priority breakdown
    result += "\n🎯 PRIORITY BREAKDOWN\n"
    for priority, count in priority_counts.items():
        emoji = priority_emojis.get(priority, '⚪')
        result += f"   {emoji} {priority}: {count}\n"

    result += f"\n{'─' * 65}\n"
    result += "📋 DETAILED TICKETS\n\n"

    # Add detailed ticket information
    for ticket in tickets:
        priority_emoji = priority_emojis.get(ticket.get('priority'), '⚪')
        status_emoji = status_emojis.get(ticket.get('status'), '❓')

        result += f"""
┌─ TICKET #{ticket.get('ticket_id')} ─────────────────────────────────────────────────┐
│ {priority_emoji} {ticket.get('priority')} Priority | {status_emoji} {ticket.get('status')} Status                           │
└─────────────────────────────────────────────────────────────────────┘

📌 Subject: {ticket.get('subject')}
📅 Created: {ticket.get('created_date')}
📅 Resolved: {ticket.get('resolved_date') or 'Not resolved'}

👤 Contact: {ticket.get('contact_name') or 'N/A'} ({ticket.get('contact_title') or 'N/A'})
📧 Email: {ticket.get('contact_email') or 'N/A'}
🔑 Primary Contact: {'Yes' if ticket.get('primary_contact') else 'No'}

📝 Description:
{ticket.get('description') or 'No description provided'}

{'─' * 65}
"""

    return result


# ============== MCP RESOURCES ==============

@mcp.resource("file:///logs/app.log", name="app_logs", description="TechNova Application Logs")
async def get_app_logs() -> str:
    """Read and return the application logs."""
    try:
        log_path = Path(__file__).parent / "logs" / "app.log"
        if not log_path.exists():
            raise ResourceError("Application log file not found")

        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            return "No application logs available"

        return content
    except Exception as e:
        raise ResourceError(f"Error reading application logs: {str(e)}")


@mcp.resource("file:///logs/customer_{customer_id}.log")
async def get_customer_logs(customer_id: str) -> str:
    """Read and return customer-specific logs."""
    try:
        log_path = Path(__file__).parent / "logs" / f"customer_{customer_id}.log"
        if not log_path.exists():
            raise ResourceError(f"Customer log file not found for {customer_id}")

        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            return f"No logs available for customer {customer_id}"

        return content
    except Exception as e:
        raise ResourceError(f"Error reading customer logs for {customer_id}: {str(e)}")


# ============== MCP PROMPTS ==============

@mcp.prompt(
    name="customer_issue_summary",
    description="Create a comprehensive customer issue summary from logs and support data",
    tags={"customer", "summary", "support"}
)
def customer_issue_summary(customer_id: str, timeframe: str = "24hours") -> str:
    """Generate a comprehensive customer issue summary prompt.

    Args:
        customer_id: The customer ID to analyze
        timeframe: Time period for analysis (e.g., "24hours", "7days", "30days")

    Returns:
        A structured prompt for creating a customer issue summary
    """
    return f"""
Please create a comprehensive customer issue summary for customer {customer_id} covering the last {timeframe}.

Analyze the following data sources:
1. Customer support tickets and their resolution status
2. Application logs showing system interactions and errors
3. Customer-specific activity logs
4. Any patterns in issues or system behavior

Format as a structured briefing document that a senior support agent can quickly understand and act upon.

Include the following sections:
- **Customer Overview**: Basic customer information and subscription details
- **Recent Activity Summary**: Key activities and interactions in the specified timeframe
- **Issues Identified**: Any problems, errors, or concerns found
- **Resolution Status**: Current status of any open issues
- **Patterns & Trends**: Any recurring issues or notable patterns
- **Recommendations**: Suggested actions for the support team
- **Priority Level**: Assessment of urgency and impact

Please ensure the summary is:
- Concise but comprehensive
- Action-oriented with clear next steps
- Prioritized by business impact
- Professional and suitable for management review
"""


if __name__ == "__main__":
    # Configure server with streamable-http transport
    # Available transports: "streamable-http", "sse", "stdio"
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8001)
