name := "MleapIM"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.1.0"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.1.0"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.1.0"
libraryDependencies += "ml.combust.mleap" %% "mleap-runtime" % "0.8.0-SNAPSHOT"
libraryDependencies += "ml.combust.mleap" %% "mleap-spark" % "0.8.0-SNAPSHOT"

