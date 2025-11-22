export interface LegalCase {
  id: string;
  title: string;
  citation: string;
  court: string;
  date: string;
  jurisdiction: string;
  summary: string;
  fullText: string;
  citations: string[];
  similarCases: string[];
  contradictions: string[];
  embedding?: number[];
  categories: string[];
  judges: string[];
}

export const mockCases: LegalCase[] = [
  {
    id: "case-001",
    title: "Smith v. Jones",
    citation: "589 U.S. 342 (2024)",
    court: "Supreme Court of the United States",
    date: "2024-11-15",
    jurisdiction: "Federal",
    summary: "The Supreme Court held that employment contracts containing mandatory arbitration clauses must explicitly disclose the waiver of class action rights. The Court found that buried or ambiguous language violates procedural due process and public policy favoring informed consent.",
    fullText: `SUPREME COURT OF THE UNITED STATES

Smith v. Jones, 589 U.S. 342 (2024)

OPINION OF THE COURT

Justice Martinez delivered the opinion of the Court.

This case concerns whether mandatory arbitration clauses in employment contracts must explicitly disclose the waiver of class action rights to be enforceable under the Federal Arbitration Act (FAA) and principles of procedural due process.

I. BACKGROUND

Petitioner Smith was employed by Jones Corporation under a standard employment agreement. The agreement contained a mandatory arbitration clause requiring all disputes to be resolved through binding arbitration. The clause included a waiver of class action rights, but this provision was embedded in dense legal language and not separately highlighted or explained.

When Smith sought to bring a class action lawsuit alleging systematic wage violations, Jones Corporation moved to compel arbitration based on the employment agreement. The District Court granted the motion, and the Court of Appeals affirmed, holding that the arbitration clause was valid under the FAA.

II. ANALYSIS

The Federal Arbitration Act generally favors the enforcement of arbitration agreements. However, this preference must be balanced against fundamental principles of contract formation, including mutual assent and informed consent. An arbitration agreement, like any contract, must be entered into knowingly and voluntarily.

Class action waivers represent a significant surrender of legal rights. They prevent employees from pooling resources to challenge systematic violations and effectively insulate employers from accountability for low-value but widespread claims. Given the magnitude of this waiver, we hold that procedural due process requires clear and conspicuous disclosure.

The employment agreement in this case failed to meet this standard. The class action waiver was buried within a lengthy arbitration clause, using technical legal terminology that would not be readily understood by an ordinary employee. There was no separate acknowledgment or signature specifically addressing this waiver. Under these circumstances, we cannot conclude that Smith knowingly and voluntarily surrendered his right to pursue class action relief.

III. HOLDING

We hold that mandatory arbitration clauses containing class action waivers must be presented in a clear, conspicuous, and separately acknowledged manner to be enforceable. The agreement in this case did not meet this standard, and therefore the arbitration clause is unenforceable with respect to class action claims.

The judgment of the Court of Appeals is REVERSED, and the case is remanded for proceedings consistent with this opinion.

It is so ordered.`,
    citations: ["Federal Arbitration Act", "AT&T Mobility v. Concepcion", "Epic Systems v. Lewis"],
    similarCases: ["case-004", "case-007"],
    contradictions: ["case-002"],
    categories: ["Employment Law", "Arbitration", "Contract Law"],
    judges: ["Justice Martinez", "Chief Justice Chen", "Justice O'Brien"],
  },
  {
    id: "case-002",
    title: "Tech Corp. v. Innovation LLC",
    citation: "234 F.3d 891 (9th Cir. 2024)",
    court: "United States Court of Appeals, Ninth Circuit",
    date: "2024-11-10",
    jurisdiction: "Federal - 9th Circuit",
    summary: "Court upheld broad arbitration clauses in employment contracts, holding that general language sufficiently puts employees on notice of class action waivers. This ruling contradicts recent Supreme Court precedent on explicit disclosure requirements.",
    fullText: `UNITED STATES COURT OF APPEALS
FOR THE NINTH CIRCUIT

Tech Corp. v. Innovation LLC

No. 23-4567

OPINION

The question before this Court is whether an arbitration clause in an employment contract must contain explicit, separately acknowledged language regarding class action waivers, or whether general arbitration language is sufficient to encompass such waivers.

FACTS

Innovation LLC required all employees to sign employment agreements containing mandatory arbitration clauses. The relevant provision stated: "All disputes arising from or relating to employment shall be resolved through binding arbitration in accordance with the rules of the American Arbitration Association."

When several employees sought to bring a class action for overtime violations, Tech Corp. (Innovation LLC's successor) moved to compel individual arbitration, arguing that the general arbitration clause encompassed all claims, including those that would otherwise be pursued on a class basis.

ANALYSIS

The Federal Arbitration Act establishes a strong federal policy favoring arbitration. While arbitration agreements must be entered into knowingly and voluntarily, we hold that detailed, itemized disclosure of every consequence of arbitration is not required.

The arbitration clause at issue clearly states that "all disputes" must be resolved through arbitration. A reasonable employee would understand that this encompasses the manner and forum of dispute resolution, including whether claims may be pursued collectively or individually. The nature of arbitration—as an individual dispute resolution mechanism—inherently suggests that class proceedings are not contemplated.

We decline to impose additional disclosure requirements beyond those established by the FAA and general contract principles. Parties are presumed to understand the basic consequences of the agreements they sign, particularly in the commercial and employment contexts.

CONCLUSION

The district court's order compelling individual arbitration is AFFIRMED.`,
    citations: ["Federal Arbitration Act", "Epic Systems v. Lewis", "AT&T Mobility v. Concepcion"],
    similarCases: ["case-001", "case-005"],
    contradictions: ["case-001"],
    categories: ["Employment Law", "Arbitration", "Contract Law"],
    judges: ["Judge Thompson", "Judge Williams", "Judge Davidson"],
  },
  {
    id: "case-003",
    title: "State v. Johnson",
    citation: "892 P.2d 456 (Cal. 2024)",
    court: "California Supreme Court",
    date: "2024-11-08",
    jurisdiction: "California",
    summary: "California Supreme Court held that evidence obtained through warrantless thermal imaging of a residence violates the Fourth Amendment and California Constitution's privacy protections, requiring suppression at trial.",
    fullText: `CALIFORNIA SUPREME COURT

State of California v. Marcus Johnson

Case No. S271234

MAJORITY OPINION by Justice Rodriguez

This case requires us to determine whether the use of thermal imaging technology to detect heat patterns emanating from a private residence, without a warrant, violates constitutional protections against unreasonable searches.

FACTS

Law enforcement received an anonymous tip that Johnson was cultivating marijuana in his home. Without obtaining a warrant, officers used a thermal imaging device from a public street to scan Johnson's residence. The scan revealed heat patterns consistent with high-intensity grow lights. Based on this thermal imaging data, combined with utility records showing high electricity usage, officers obtained a search warrant and discovered an indoor marijuana cultivation operation.

Johnson moved to suppress the evidence, arguing that the warrantless thermal imaging violated his Fourth Amendment rights and the privacy protections guaranteed by the California Constitution.

LEGAL FRAMEWORK

The Fourth Amendment and California Constitution Article I, Section 13 protect individuals against unreasonable searches and seizures. The United States Supreme Court has established that warrantless searches are presumptively unreasonable, subject to a few specifically established exceptions.

In Kyllo v. United States, the Supreme Court held that the use of sense-enhancing technology to obtain information about the interior of a home that could not otherwise be obtained without physical intrusion constitutes a Fourth Amendment search. The Court emphasized that the home occupies a special position in Fourth Amendment jurisprudence, and that individuals have a legitimate expectation of privacy in the details of their home that are not exposed to public view.

ANALYSIS

Thermal imaging technology reveals information about the interior of a home by detecting and measuring heat radiation. While the technology does not provide visual images of the home's interior, it discloses details about activities within the home that would not be observable through normal sensory perception from a public vantage point.

The heat patterns detected by thermal imaging can reveal intimate details of home life, including when residents are present, which rooms are occupied, what appliances are being used, and patterns of daily activity. This level of intrusion into the private sphere of the home requires judicial authorization through the warrant process.

The State argues that Johnson had no reasonable expectation of privacy in the heat emanating from his home because it escapes into public space. We reject this argument. The fact that certain byproducts of home activities escape the physical confines of the residence does not eliminate the expectation of privacy in the underlying activities generating those byproducts.

California's constitutional privacy protections are broader than those provided by the Fourth Amendment. While Kyllo addressed thermal imaging specifically in the context of high-intensity lamp detection, we hold that California law extends stronger privacy protections to all uses of sense-enhancing technology that reveal details about the home's interior.

HOLDING

We hold that the warrantless use of thermal imaging technology to scan a private residence violates both the Fourth Amendment and the privacy provisions of the California Constitution. The evidence obtained as a result of this unconstitutional search, including all derivative evidence, must be suppressed.

The judgment of the Court of Appeals is AFFIRMED.`,
    citations: ["Kyllo v. United States", "California Constitution Article I §13", "Fourth Amendment"],
    similarCases: ["case-006", "case-008"],
    contradictions: [],
    categories: ["Criminal Law", "Search and Seizure", "Privacy", "Fourth Amendment"],
    judges: ["Justice Rodriguez", "Justice Kim", "Justice Anderson"],
  },
  {
    id: "case-004",
    title: "DataCorp v. Privacy Advocates",
    citation: "445 F. Supp. 3d 234 (S.D.N.Y. 2024)",
    court: "United States District Court, Southern District of New York",
    date: "2024-10-28",
    jurisdiction: "Federal - SDNY",
    summary: "District court found that class action waivers in consumer contracts of adhesion are procedurally unconscionable when they effectively prevent consumers from vindicating their rights due to the low value of individual claims.",
    fullText: `UNITED STATES DISTRICT COURT
SOUTHERN DISTRICT OF NEW YORK

DataCorp v. Privacy Advocates, et al.

Case No. 1:23-cv-09876

MEMORANDUM OPINION AND ORDER

DataCorp moves to compel arbitration and dismiss this putative class action alleging violations of privacy statutes. The central question is whether the arbitration clause in DataCorp's Terms of Service, which includes a class action waiver, is enforceable under federal and state law.

BACKGROUND

Privacy Advocates brought this action on behalf of a class of millions of users whose personal data was allegedly collected and sold without proper consent, in violation of state privacy statutes. The alleged damages per user range from $100 to $750. DataCorp's Terms of Service, which users must accept to use the service, contains a mandatory arbitration clause with a class action waiver.

LEGAL STANDARD

Under the Federal Arbitration Act, arbitration agreements are generally enforceable. However, courts retain the authority to invalidate arbitration provisions that are unconscionable under generally applicable state contract law principles.

New York law recognizes both procedural and substantive unconscionability. Procedural unconscionability focuses on the contract formation process, including unequal bargaining power and the lack of meaningful choice. Substantive unconscionability examines whether the contract terms are unreasonably favorable to one party.

ANALYSIS

Procedural Unconscionability

DataCorp's Terms of Service is a classic contract of adhesion. Users have no ability to negotiate its terms; they must accept or forgo the service entirely. The arbitration clause and class action waiver are buried deep within a lengthy document written in dense legal language. There is no evidence that users are specifically directed to these provisions or that they understand the significance of waiving class action rights.

While contracts of adhesion are not per se unenforceable, the complete absence of bargaining power, combined with the inconspicuous presentation of critical terms, establishes procedural unconscionability.

Substantive Unconscionability

The substantive unconscionability analysis requires examining whether the class action waiver effectively denies plaintiffs a forum to vindicate their statutory rights. Given that individual claims in this case range from $100 to $750, and that the cost of individual arbitration likely exceeds the value of any individual recovery, the class action waiver has the practical effect of insulating DataCorp from liability.

The vindication of rights doctrine, though narrowly applied, recognizes that arbitration agreements cannot be used to preclude the effective vindication of federal or state statutory rights. Here, the combination of small individual claims, high arbitration costs, and the class action waiver creates a situation where no rational plaintiff would pursue an individual claim.

This stands in contrast to cases where individual claims are sufficiently large to justify individual arbitration, or where the arbitration agreement includes fee-shifting or other provisions that make individual arbitration economically viable.

CONCLUSION

The Court finds that the class action waiver in DataCorp's arbitration clause is both procedurally and substantively unconscionable as applied to this case. The motion to compel arbitration is DENIED with respect to the class action waiver. The arbitration clause may be enforced for any individual claims that plaintiffs wish to arbitrate, but it cannot be used to bar class proceedings.

SO ORDERED.`,
    citations: ["Federal Arbitration Act", "AT&T Mobility v. Concepcion", "American Express v. Italian Colors"],
    similarCases: ["case-001", "case-007"],
    contradictions: ["case-002"],
    categories: ["Consumer Protection", "Arbitration", "Privacy Law", "Class Actions"],
    judges: ["Judge Martinez"],
  },
];

export const knowledgeGraphData = {
  nodes: [
    { id: "case-001", type: "case", label: "Smith v. Jones", level: "Supreme Court" },
    { id: "case-002", type: "case", label: "Tech Corp v. Innovation", level: "9th Circuit" },
    { id: "case-003", type: "case", label: "State v. Johnson", level: "State Supreme" },
    { id: "case-004", type: "case", label: "DataCorp v. Privacy", level: "District" },
    { id: "court-scotus", type: "court", label: "US Supreme Court" },
    { id: "court-9th", type: "court", label: "9th Circuit" },
    { id: "court-cal", type: "court", label: "California Supreme" },
    { id: "judge-martinez", type: "judge", label: "Justice Martinez" },
    { id: "judge-thompson", type: "judge", label: "Judge Thompson" },
    { id: "statute-faa", type: "statute", label: "Federal Arbitration Act" },
    { id: "statute-4th", type: "statute", label: "Fourth Amendment" },
  ],
  edges: [
    { id: "e1", source: "case-001", target: "court-scotus", type: "DECIDED_BY" },
    { id: "e2", source: "case-002", target: "court-9th", type: "DECIDED_BY" },
    { id: "e3", source: "case-003", target: "court-cal", type: "DECIDED_BY" },
    { id: "e4", source: "case-001", target: "case-004", type: "SIMILAR_TO", similarity: 0.89 },
    { id: "e5", source: "case-001", target: "case-002", type: "CONTRADICTS", conflict: "arbitration disclosure" },
    { id: "e6", source: "case-001", target: "statute-faa", type: "CITES" },
    { id: "e7", source: "case-002", target: "statute-faa", type: "CITES" },
    { id: "e8", source: "case-003", target: "statute-4th", type: "CITES" },
    { id: "e9", source: "case-001", target: "judge-martinez", type: "AUTHORED_BY" },
    { id: "e10", source: "case-002", target: "judge-thompson", type: "AUTHORED_BY" },
  ],
};
