# Troubleshooting Guide

## Common Issues

### Cursor AI Agent Not Connecting

#### Symptoms
- Cursor AI agent stops responding
- AI features not working
- Agent appears disconnected or unresponsive

#### Solution
For macOS users, clearing the Cursor application data and cache can resolve connection issues:

```bash
rm -rf ~/Library/Application\ Support/Cursor
rm -rf ~/Library/Caches/Cursor
```

After running these commands:
1. Restart the Cursor application
2. The AI agent should now connect properly

#### Why This Works
This solution clears out any corrupted application state or cached data that might be preventing the AI agent from establishing a proper connection.

## Other Issues

If you encounter other issues, please:
1. Check your internet connection
2. Ensure you're running the latest version of Cursor
3. If problems persist, report the issue on our [GitHub repository](https://github.com/zachbovaird/UHG-Library/issues) 