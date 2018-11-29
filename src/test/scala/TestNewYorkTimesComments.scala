object TestNewYorkTimesComments {
  def main(args: Array[String]): Unit = {
    //print(NewYorkTimesComments.preprocessText("They Can Hit a Ball 400 Feet. But Play Catch? That’s Tricky."))
    //NewYorkTimesComments.parseAuthor("By EZRA LEVIN, LEAH GREENBERG and ANGEL PADILLA").foreach(println(_))
    NewYorkTimesComments.parseKeywords("['Taxation', 'Trump, Donald J', 'Koch, Charles G', 'Koch, David H', 'Americans for Prosperity', 'Club for Growth', 'Koch Industries Inc', 'Republican Party', 'United States Politics and Government', 'House of Representatives', 'International Trade and World Market']").foreach(println(_))
  }
}
