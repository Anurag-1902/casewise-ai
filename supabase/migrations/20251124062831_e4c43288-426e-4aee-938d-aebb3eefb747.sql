-- Create legal cases table
CREATE TABLE public.legal_cases (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  case_id TEXT UNIQUE NOT NULL,
  name TEXT NOT NULL,
  name_abbreviation TEXT,
  citation TEXT,
  court TEXT,
  decision_date DATE,
  jurisdiction TEXT,
  url TEXT,
  frontend_url TEXT,
  preview TEXT[],
  docket_number TEXT,
  first_page TEXT,
  last_page TEXT,
  analysis JSONB,
  volume JSONB,
  reporter JSONB,
  court_info JSONB,
  citations JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Enable RLS
ALTER TABLE public.legal_cases ENABLE ROW LEVEL SECURITY;

-- Allow everyone to read cases (public data)
CREATE POLICY "Anyone can read legal cases"
ON public.legal_cases
FOR SELECT
TO public
USING (true);

-- Create updated_at trigger
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_legal_cases_updated_at
BEFORE UPDATE ON public.legal_cases
FOR EACH ROW
EXECUTE FUNCTION public.update_updated_at_column();

-- Create index for faster searches
CREATE INDEX idx_legal_cases_name ON public.legal_cases(name);
CREATE INDEX idx_legal_cases_jurisdiction ON public.legal_cases(jurisdiction);
CREATE INDEX idx_legal_cases_decision_date ON public.legal_cases(decision_date);
CREATE INDEX idx_legal_cases_case_id ON public.legal_cases(case_id);