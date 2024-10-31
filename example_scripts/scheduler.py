#!/usr/bin/env python3
import os
import argparse
from crontab import CronTab

def clear_crontab():
    """Remove all cron jobs for the current user"""
    cron = CronTab(user=True)
    cron.remove_all()
    cron.write()
    print("Cleared all cron jobs")

def remove_jobs_by_comment(comment: str):
    """Remove all cron jobs with the specified comment"""
    cron = CronTab(user=True)
    removed = cron.remove_all(comment=comment)
    cron.write()
    print(f"Removed {removed} job(s) with comment '{comment}'")

def add_cron_job(timing: str, script_path: str, python_exec: str, 
                 log_file: str = None, comment: str = "trading_bot", 
                 overwrite: bool = False):
    """
    Add a cron job for the specified script
    
    Args:
        timing: Cron timing string (e.g., "30 9 * * 1-5" for 9:30 AM on weekdays)
        script_path: Full path to the Python script
        python_exec: Path to Python executable
        log_file: Optional path to log file for redirecting output
        comment: Comment identifier for the cron job
        overwrite: If True, removes all existing jobs with the same comment before adding new one
    """
    # Initialize crontab for the current user
    cron = CronTab(user=True)
    
    # If overwrite is True, remove existing jobs with the same comment
    if overwrite:
        cron.remove_all(comment=comment)
    
    # Construct the command
    command = f'cd {os.path.dirname(script_path)} && {python_exec} "{script_path}"'
    if log_file:
        command += f' >> "{log_file}" 2>&1'
    
    # Create new cron job
    job = cron.new(command=command, comment=comment)
    job.setall(timing)
    
    # Write the jobs to crontab
    cron.write()
    return job

def print_help():
    """Print detailed help information and usage examples"""
    help_text = """
Scheduler Help
=============

Description:
    This script sets up cron jobs for trading-related scripts using the system's crontab.

Common Cron Timing Patterns:
    - "30 9 * * 1-5"    : Run at 9:30 AM on weekdays
    - "0 10-16 * * 1-5" : Run every hour from 10 AM to 4 PM on weekdays
    - "*/30 9-16 * * 1-5": Run every 30 minutes between 9 AM and 4 PM on weekdays

Examples:
    1. Set up fetch_option_chain.py to run at market open (9:30 AM ET) on weekdays:
       ./scheduler.py --timing "30 9 * * 1-5" --script "/path/to/fetch_option_chain.py" --python "/usr/bin/python3" --log "fetch.log"

    2. Set up bot_status.py to run hourly during market hours:
       ./scheduler.py --timing "0 10-16 * * 1-5" --script "/path/to/bot_status.py" --python "/usr/bin/python3" --log "bot_status.log"

    3. Overwrite existing cron jobs with the same comment:
       ./scheduler.py --timing "*/30 9-16 * * 1-5" --script "/path/to/script.py" --python "/usr/bin/python3" --overwrite

    4. Clear all cron jobs for the current user:
       ./scheduler.py --clear

    5. Remove all jobs with a specific comment:
       ./scheduler.py --remove-comment "trading_bot"

Note:
    - The script will cd into the script's directory before execution
    - Log files will contain both stdout and stderr
    - Use --overwrite to replace existing jobs with the same comment
    - Times are in the system's timezone, adjust accordingly for market hours
    """
    print(help_text)

def main():
    parser = argparse.ArgumentParser(
        description='Set up cron jobs for trading scripts',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--timing',
                       help='Cron timing string (e.g., "30 9 * * 1-5" for 9:30 AM on weekdays)')
    parser.add_argument('--help-examples', action='store_true', 
                       help='Show detailed help with usage examples')
    parser.add_argument('--script', help='Path to the Python script to run')
    parser.add_argument('--python', help='Path to Python executable')
    parser.add_argument('--log', help='Path to log file (optional)')
    parser.add_argument('--comment', default='trading_bot', help='Comment identifier for the cron job')
    parser.add_argument('--overwrite', action='store_true', help='Remove existing jobs with same comment')
    parser.add_argument('--clear', action='store_true', help='Clear all cron jobs for current user')
    parser.add_argument('--remove-comment', help='Remove all jobs with specified comment')
    
    args = parser.parse_args()
    
    if args.help_examples:
        print_help()
        return
    
    if args.clear:
        clear_crontab()
        return

    if args.remove_comment:
        remove_jobs_by_comment(args.remove_comment)
        return
        
    # Validate required arguments when not using --help-examples, --clear, or --remove-comment
    if not all([args.timing, args.script, args.python]):
        parser.error("--timing, --script, and --python are required unless using --help-examples or --clear")
        
    # Add the cron job
    job = add_cron_job(
        timing=args.timing,
        script_path=args.script,
        python_exec=args.python,
        log_file=args.log,
        comment=args.comment,
        overwrite=args.overwrite
    )
    
    print(f"Cron job has been set up successfully!")
    print(f"Added job: {job}")
    
    # Show all current jobs with the same comment
    cron = CronTab(user=True)
    print(f"\nAll current jobs with comment '{args.comment}':")
    for j in cron.find_comment(args.comment):
        print(j)

if __name__ == "__main__":
    main()
