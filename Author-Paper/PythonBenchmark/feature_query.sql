WITH 
LastPubYear As(
	SELECT AuthorId, max(Year) as year 
	FROM PaperAuthor 
	LEFT OUTER JOIN Paper ON paperid=id
	GROUP BY AuthorId
),
AuthorJournalCounts AS (
    SELECT AuthorId, JournalId, Count(*) AS Count
    FROM PaperAuthor pa
    LEFT OUTER JOIN Paper p on pa.PaperId=p.Id
    GROUP BY AuthorId, JournalId),
AuthorConferenceCounts AS (
    SELECT AuthorId, ConferenceId, Count(*) AS Count
    FROM PaperAuthor pa
    LEFT OUTER JOIN Paper p on pa.PaperId=p.Id
    GROUP BY AuthorId, ConferenceId),
AuthorPaperCounts AS (
    SELECT AuthorId, Count(*) AS Count
    FROM PaperAuthor
    GROUP BY AuthorId),
PaperAuthorCounts AS (
    SELECT PaperId, Count(*) AS Count
    FROM PaperAuthor
    GROUP BY PaperId),
PapersInSameYear AS (
	SELECT AuthorId, Year, Count(*) As Count
	FROM PaperAuthor pa
	LEFT OUTER JOIN Paper p on pa.PaperId=p.Id
	GROUP BY AuthorId, Year),
SumPapersWithCoAuthors AS (
    WITH CoAuthors AS (
        SELECT pa1.AuthorId Author1, 
               pa2.AuthorId Author2, 
               COUNT(*) AS NumPapersTogether
        FROM PaperAuthor pa1,
             PaperAuthor pa2
        WHERE pa1.PaperId=pa2.PaperId
          AND pa1.AuthorId != pa2.AuthorId
          AND pa1.AuthorId IN (
              SELECT DISTINCT AuthorId
              FROM ##DataTable##)
        GROUP BY pa1.AuthorId, pa2.AuthorId)
    SELECT t.AuthorId,
           t.PaperId, 
           SUM(NumPapersTogether) AS Sum
    FROM ##DataTable## t
    LEFT OUTER JOIN PaperAuthor pa ON t.PaperId=pa.PaperId
    LEFT OUTER JOIN CoAuthors ca ON ca.Author2=pa.AuthorId
    WHERE pa.AuthorId != t.AuthorId
      AND ca.Author1 = t.AuthorId
    GROUP BY t.AuthorId, t.PaperId
)
SELECT t.AuthorId,
       t.PaperId,
       ajc.Count As NumSameJournal, 
       acc.Count AS NumSameConference,
       apc.Count AS NumPapersWithAuthor,
       pac.Count AS NumAuthorsWithPaper,
       pin.Count As NumPapersInSameYear,
       CASE WHEN (lpy.year <= 0 or p.year <= 0) THEN 0
		ELSE lpy.year - p.year
       END AS DiffLastPubYear,
       CASE WHEN coauth.Sum > 0 THEN cast(coauth.Sum as integer)
            ELSE 0 
       END AS SumPapersWithCoAuthors
FROM ##DataTable## t
LEFT OUTER JOIN Paper p ON t.PaperId=p.Id
LEFT OUTER JOIN AuthorJournalCounts ajc
    ON ajc.AuthorId=t.AuthorId
  AND ajc.JournalId = p.JournalId
LEFT OUTER JOIN AuthorConferenceCounts acc
    ON acc.AuthorId=t.AuthorId
   AND acc.ConferenceId = p.ConferenceId
LEFT OUTER JOIN AuthorPaperCounts apc
    ON apc.AuthorId=t.AuthorId
LEFT OUTER JOIN PaperAuthorCounts pac
    ON pac.PaperId=t.PaperId
LEFT OUTER JOIN PapersInSameYear pin
	ON pin.AuthorId=t.AuthorId
   AND pin.Year=p.Year
LEFT OUTER JOIN LastPubYear lpy 
	ON lpy.AuthorId = t.AuthorId
LEFT OUTER JOIN SumPapersWithCoAuthors coauth
    ON coauth.AuthorId=t.AuthorId
   AND coauth.PaperId=t.PaperId 
ORDER BY t.authorid, t.paperid
