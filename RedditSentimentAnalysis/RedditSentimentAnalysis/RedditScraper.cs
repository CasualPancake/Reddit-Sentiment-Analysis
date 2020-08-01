using System;
using System.Collections.Generic;
using System.Linq;
using RedditSharp;

public class RedditScraper
{
	private Reddit reddit;

	public RedditScraper()
	{
			reddit = new Reddit();
	}

	public List<string> ScrapeCommentsSinglePost(string subredditLink)
	{
		var subreddit = reddit.GetSubreddit(subredditLink);
		List<string> commentList = new List<string>();

		foreach (var post in subreddit.Hot.Take(1))
		{
			foreach (var comment in post.Comments.Take(5))
			{
				commentList.Add(comment.Body);
				Console.WriteLine(comment.Body);
			}
		}

		return commentList;
	}
}
